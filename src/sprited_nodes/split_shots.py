#!/usr/bin/env python3
"""
split_shots.py — Split an input clip into one file per shot using
PySceneDetect + FFmpeg.  Supports MP4 or animated WebP in, MP4/WebP out.
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple
from inspect import cleandoc

# ── PySceneDetect -----------------------------------------------------------
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    from scenedetect.frame_timecode import FrameTimecode
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: PySceneDetect not available. Install with: pip install scenedetect")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

# ---------------------------------------------------------------------------
# Scene detection helpers
# ---------------------------------------------------------------------------

def detect_scenes(
    video: Path, *, detector_cls, threshold: float, min_len: int
) -> List[Tuple]:
    """Return [(start, end_exclusive), …] in FrameTimecodes."""
    if not SCENEDETECT_AVAILABLE:
        raise RuntimeError("PySceneDetect is required but not installed. Install with: pip install scenedetect")
    
    vm = VideoManager([str(video)])
    sm = SceneManager()
    sm.add_detector(detector_cls(threshold=threshold, min_scene_len=min_len))
    try:
        vm.start()
        sm.detect_scenes(frame_source=vm)
        return sm.get_scene_list()
    finally:
        vm.release()

# ---------------------------------------------------------------------------
# WebP → temp MP4 shim (loss-less x264)
# ---------------------------------------------------------------------------

def webp_to_tmp_mp4(src: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="splitshots_"))
    dst = tmp_dir / f"{src.stem}.mp4"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst),
    ]
    if subprocess.call(cmd) == 0 and dst.stat().st_size:
        return dst

    # Fallback: webpmux → PNGs → MP4
    if not shutil.which("webpmux"):
        raise RuntimeError("Need 'webpmux' or an FFmpeg with WebP support.")
    frames = tmp_dir / "frames"; frames.mkdir()
    subprocess.check_call(["webpmux", "-dump", str(src),
                           "-o", str(frames / "frame.png")])
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", "30", "-pattern_type", "glob",
        "-i", str(frames / "frame*.png"),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst),
    ])
    for p in glob.glob(str(frames / "frame*.png")): os.remove(p)
    return dst

# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------

def encode_cmd(fmt: str, src: Path, start_ts: str, n_frames: int, dst: Path,
               *, reencode: bool) -> List[str]:
    """Return FFmpeg cmd that copies/encodes exactly n_frames starting @ start."""
    if fmt == "mp4" and not reencode:
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", start_ts, "-i", str(src),
            "-frames:v", str(n_frames),
            "-c", "copy", str(dst),
        ]
    common = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", start_ts, "-i", str(src),
        "-frames:v", str(n_frames)
    ]
    if fmt == "mp4":   # re-encode loss-less H.264
        common += ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
                   "-pix_fmt", "yuv444p"]
    elif fmt == "webp":
        common += ["-an", "-vcodec", "libwebp", "-lossless", "1",
                   "-preset", "default", "-loop", "0", "-vsync", "0"]
    else:
        raise ValueError(fmt)
    return common + [str(dst)]

# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def split_scenes(src: Path, scenes, out_dir: Path, *, fmt: str, reencode: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Detected {len(scenes)} shots.")
    for idx, (start, end_excl) in enumerate(scenes):
        n = end_excl.get_frames() - start.get_frames()  # <= exact length
        if n <= 0: continue
        dst = out_dir / f"shot-{idx:03}.{fmt}"
        cmd = encode_cmd(fmt, src, start.get_timecode(), n, dst,
                         reencode=reencode)
        subprocess.check_call(cmd)
        print(f"    ↳ {dst.name}  ({start.get_timecode()} × {n} frames)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Split a clip into per-shot files.")
    p.add_argument("video", type=Path, help="Source .mp4 or animated .webp")
    p.add_argument("--out", type=Path, default=Path("./shots"))
    p.add_argument("--detector", choices=["content", "adaptive"],
                   default="content")
    p.add_argument("--threshold", type=float, default=8.0)
    p.add_argument("--min-scene-len", type=int, default=15)
    p.add_argument("--seconds-per-shot", type=float)
    p.add_argument("--format", choices=["mp4", "webp"], default="mp4")
    p.add_argument("--reencode", action="store_true", default=True)
    args = p.parse_args()

    work = args.video
    tmpdir = None
    if work.suffix.lower() == ".webp":
        print("[i] Converting WebP → temp MP4 for detection…")
        work = webp_to_tmp_mp4(work)
        tmpdir = work.parent

    if args.seconds_per_shot:                       # manual chunking
        import cv2
        cap = cv2.VideoCapture(str(work))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = int(args.seconds_per_shot * fps)
        scenes = []
        for f0 in range(0, tf, step):
            scenes.append((FrameTimecode(f0, fps),
                           FrameTimecode(min(f0 + step, tf), fps)))
        cap.release()
    else:
        Det = ContentDetector if args.detector == "content" else AdaptiveDetector
        scenes = detect_scenes(work, detector_cls=Det,
                               threshold=args.threshold,
                               min_len=args.min_scene_len)

    if not scenes:
        print("[!] No cuts detected.")
    else:
        split_scenes(work, scenes, args.out, fmt=args.format,
                     reencode=args.reencode)

    if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class VideoShotSplitter:
    """
    A ComfyUI node that splits a video into individual shots using scene detection.
    
    Takes a video file path as input and returns a list of file paths to the 
    individual shot video files.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the video shot splitter node.
        
        Returns:
            dict: Configuration for input fields
        """
        return {
            "required": {
                "file_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Path to the input video file (MP4)"
                }),
                "detector": (["content", "adaptive"], {
                    "default": "content",
                    "tooltip": "Scene detection algorithm to use"
                }),
                "threshold": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Detection sensitivity threshold"
                }),
                "min_scene_len": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Minimum scene length in frames"
                }),
                "output_format": (["mp4", "webp"], {
                    "default": "mp4",
                    "tooltip": "Output format for shot files"
                }),
                "reencode": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Whether to re-encode (slower but more compatible)"
                })
            },
            "optional": {
                "seconds_per_shot": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Manual chunking: split every N seconds (0 = auto detection)"
                }),
                "output_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Output directory (empty = auto-generate)"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_file_paths",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "split_video_shots"
    OUTPUT_NODE = False
    CATEGORY = "Video/Edit"

    def split_video_shots(self, file_path, detector, threshold, min_scene_len, 
                         output_format, reencode, seconds_per_shot=0.0, output_dir=""):
        """
        Split video into shots and return list of file paths
        
        Args:
            file_path (str): Path to input video file
            detector (str): Scene detection algorithm ("content" or "adaptive")
            threshold (float): Detection sensitivity threshold
            min_scene_len (int): Minimum scene length in frames
            output_format (str): Output format ("mp4" or "webp")
            reencode (str): Whether to re-encode ("true" or "false")
            seconds_per_shot (float): Manual chunking interval (0 = auto detection)
            output_dir (str): Output directory path
            
        Returns:
            tuple: (comma-separated list of shot file paths,)
        """
        try:
            # Check dependencies
            if seconds_per_shot == 0.0 and not SCENEDETECT_AVAILABLE:
                raise RuntimeError("PySceneDetect is required for automatic scene detection. Install with: pip install scenedetect")
            
            if seconds_per_shot > 0.0 and not CV2_AVAILABLE:
                raise RuntimeError("OpenCV is required for manual chunking. Install with: pip install opencv-python")
            
            # Validate input file
            video_path = Path(file_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            # Set up output directory
            if not output_dir:
                output_dir = video_path.parent / f"{video_path.stem}_shots"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle WebP input by converting to temp MP4
            work_path = video_path
            tmpdir = None
            if video_path.suffix.lower() == ".webp":
                print("[ComfyUI] Converting WebP → temp MP4 for detection…")
                work_path = webp_to_tmp_mp4(video_path)
                tmpdir = work_path.parent
            
            # Detect scenes
            if seconds_per_shot > 0.0:  # Manual chunking
                cap = cv2.VideoCapture(str(work_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = int(seconds_per_shot * fps)
                scenes = []
                for f0 in range(0, tf, step):
                    scenes.append((FrameTimecode(f0, fps),
                                   FrameTimecode(min(f0 + step, tf), fps)))
                cap.release()
            else:  # Automatic scene detection
                Det = ContentDetector if detector == "content" else AdaptiveDetector
                scenes = detect_scenes(work_path, detector_cls=Det,
                                       threshold=threshold, min_len=min_scene_len)
            
            if not scenes:
                print("[ComfyUI] No cuts detected.")
                return ("",)
            
            # Split scenes and collect output paths
            reencode_bool = (reencode == "true")
            shot_paths = []
            
            print(f"[ComfyUI] Detected {len(scenes)} shots.")
            for idx, (start, end_excl) in enumerate(scenes):
                n = end_excl.get_frames() - start.get_frames()
                if n <= 0:
                    continue
                    
                dst = output_dir / f"shot-{idx:03}.{output_format}"
                cmd = encode_cmd(output_format, work_path, start.get_timecode(), 
                               n, dst, reencode=reencode_bool)
                subprocess.check_call(cmd)
                shot_paths.append(str(dst))
                print(f"[ComfyUI] → {dst.name} ({start.get_timecode()} × {n} frames)")
            
            # Clean up temp directory if needed
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            
            # Return comma-separated list of paths
            result = ",".join(shot_paths)
            print(f"[ComfyUI] Split complete. Generated {len(shot_paths)} shot files.")
            return (result,)
            
        except Exception as e:
            print(f"[ComfyUI] Error splitting video shots: {str(e)}")
            return ("",)
