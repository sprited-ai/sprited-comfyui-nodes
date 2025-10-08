#!/usr/bin/env python3
"""
VideoShotSplitterV2 ─────────────────────────────────────────────────────────
Split an input video (from ComfyUI "Load Video") into individual shots and
return a list of VIDEO objects, ready for further processing.

Enhanced version that guarantees at least K shots by adaptively adjusting
the scene detection threshold. If scene detection fails to find enough shots,
it progressively lowers the threshold to be more sensitive, and falls back
to fixed-length splitting as a last resort.
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile, glob
from pathlib import Path
from typing import List, Tuple
from inspect import cleandoc
from datetime import datetime

# ---------------------------------------------------------------------------
# ComfyUI video input type
# ---------------------------------------------------------------------------
from comfy_api.input.video_types import VideoInput
from comfy_api.input_impl import VideoFromFile

# ── optional deps -----------------------------------------------------------
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    from scenedetect.frame_timecode import FrameTimecode
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── helpers ─────────────────────────────────────────────────────────────────
def detect_scenes(video: Path, *, detector_cls, threshold: float, min_len: int):
    vm = VideoManager([str(video)])
    sm = SceneManager()
    sm.add_detector(detector_cls(threshold=threshold, min_scene_len=min_len))
    try:
        vm.start()
        sm.detect_scenes(frame_source=vm)
        return sm.get_scene_list()
    finally:
        vm.release()

def detect_scenes_adaptive_k(video: Path, *, detector_cls, initial_threshold: float,
                           min_len: int, min_shots: int) -> List[Tuple]:
    """
    Adaptively detect scenes to ensure at least min_shots are found.
    Progressively lowers threshold until enough scenes are detected.
    """
    threshold = initial_threshold
    min_threshold = 0.5  # Don't go below this threshold
    threshold_multiplier = 0.7  # How much to reduce threshold each iteration
    max_iterations = 10  # Prevent infinite loops

    for iteration in range(max_iterations):
        print(f"[VideoShotSplitterV2] Trying scene detection with threshold={threshold:.1f}")
        scenes = detect_scenes(video, detector_cls=detector_cls,
                             threshold=threshold, min_len=min_len)

        print(f"[VideoShotSplitterV2] Found {len(scenes)} scenes")

        if len(scenes) >= min_shots:
            print(f"[VideoShotSplitterV2] ✓ Found {len(scenes)} >= {min_shots} scenes with threshold {threshold:.1f}")
            return scenes

        if threshold <= min_threshold:
            print(f"[VideoShotSplitterV2] ⚠ Reached minimum threshold {min_threshold}, giving up on scene detection")
            break

        threshold *= threshold_multiplier
        threshold = max(threshold, min_threshold)  # Don't go below minimum

    print(f"[VideoShotSplitterV2] ⚠ Scene detection failed to find {min_shots} scenes, will use fallback")
    return []

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using OpenCV."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV required to get video duration.")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count / fps

def webp_to_tmp_mp4(src: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="splitshots_"))
    dst = tmp_dir / f"{src.stem}.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-c:v", "libx264", "-crf", "0",
        "-pix_fmt", "yuv444p", "-movflags", "+faststart", str(dst)
    ]
    if subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size:
        return dst
    if not shutil.which("webpmux"):
        raise RuntimeError("Need 'webpmux' or FFmpeg built with WebP.")
    frames = tmp_dir / "frames"; frames.mkdir()
    subprocess.check_call(["webpmux", "-dump", str(src), "-o", str(frames / "frm.png")])
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", "30", "-pattern_type", "glob", "-i", str(frames / "frm*.png"),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst)
    ])
    for p in glob.glob(str(frames / "frm*.png")): os.remove(p)
    return dst

def encode_cmd(fmt: str, src: Path, start_ts: str, n_frames: int,
               dst: Path, *, reencode: bool, overwrite: bool = True) -> List[str]:
    base_args = ["-hide_banner", "-loglevel", "error"]
    if overwrite:
        base_args.append("-y")

    if fmt == "mp4" and not reencode:
        return ["ffmpeg"] + base_args + [
                "-ss", start_ts, "-i", str(src),
                "-frames:v", str(n_frames), "-c", "copy", str(dst)]
    cmd = ["ffmpeg"] + base_args + [
           "-ss", start_ts, "-i", str(src), "-frames:v", str(n_frames)]
    if fmt == "mp4":
        cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
                "-pix_fmt", "yuv444p"]
    elif fmt == "webp":
        cmd += ["-an", "-vcodec", "libwebp", "-lossless", "1",
                "-preset", "default", "-loop", "0", "-vsync", "0"]
    else:
        raise ValueError(fmt)
    return cmd + [str(dst)]

# ── node ────────────────────────────────────────────────────────────────────
class VideoShotSplitterV2:
    def __init__(self):
        self._temp_sources: list[Path] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "min_shots": ("INT", {"default": 3, "min": 1, "max": 100}),
                "detector": (["content", "adaptive"], {"default": "content"}),
                "threshold": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50}),
                "min_scene_len": ("INT", {"default": 15, "min": 1, "max": 300}),
                "output_format": (["mp4", "webp"], {"default": "mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seconds_per_shot": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0
                }),
                "output_dir": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "unique_filenames": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)          # list[VideoFromFile]
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("shot_videos",)
    FUNCTION      = "split"
    CATEGORY      = "Video/Edit"
    DESCRIPTION   = cleandoc(__doc__)

    # ── main ────────────────────────────────────────────────────────────────
    def split(self, video, min_shots, detector, threshold, min_scene_len,
              output_format, reencode,
              seconds_per_shot=0.0, output_dir="", overwrite=True, unique_filenames=True):

        try:
            src_path = self._extract_video_path(video)
            if not src_path:
                raise RuntimeError("Unable to resolve video file path.")
            src_path = Path(src_path)

            out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_shots"
            out_dir.mkdir(parents=True, exist_ok=True)

            # WebP → temp MP4
            if src_path.suffix.lower() == ".webp":
                work_path = webp_to_tmp_mp4(src_path)
                tmp_dir   = work_path.parent
            else:
                work_path, tmp_dir = src_path, None

            # Build scene list with adaptive approach
            scenes = []

            if seconds_per_shot > 0:
                # User specified fixed duration - use that
                print(f"[VideoShotSplitterV2] Using fixed duration: {seconds_per_shot}s per shot")
                scenes = self._fixed_length_scenes(work_path, seconds_per_shot)
            else:
                # Try adaptive scene detection first
                if not SCENEDETECT_AVAILABLE:
                    raise RuntimeError("PySceneDetect not installed.")

                Det = ContentDetector if detector == "content" else AdaptiveDetector
                scenes = detect_scenes_adaptive_k(
                    work_path,
                    detector_cls=Det,
                    initial_threshold=threshold,
                    min_len=min_scene_len,
                    min_shots=min_shots
                )

                # Fallback to fixed-length if adaptive detection failed
                if not scenes or len(scenes) < min_shots:
                    print(f"[VideoShotSplitterV2] Falling back to fixed-length splitting for {min_shots} shots")
                    try:
                        duration = get_video_duration(work_path)
                        seconds_per_shot_calc = duration / min_shots
                        scenes = self._fixed_length_scenes(work_path, seconds_per_shot_calc)
                        print(f"[VideoShotSplitterV2] Created {len(scenes)} fixed-length shots ({seconds_per_shot_calc:.1f}s each)")
                    except Exception as e:
                        print(f"[VideoShotSplitterV2] Fixed-length fallback failed: {e}")
                        return ([],)

            if not scenes:
                print("[VideoShotSplitterV2] No shots could be created.")
                return ([],)

            # Ensure we have at least min_shots
            if len(scenes) < min_shots:
                print(f"[VideoShotSplitterV2] WARNING: Only created {len(scenes)} shots, less than requested {min_shots}")

            # Process scenes into video files
            shot_videos = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if unique_filenames else ""

            for idx, (start, end) in enumerate(scenes):
                n = end.get_frames() - start.get_frames()
                if n <= 0: continue

                # Create filename with optional timestamp for uniqueness
                if unique_filenames:
                    filename = f"shot-{timestamp}-{idx:03}.{output_format}"
                else:
                    filename = f"shot-{idx:03}.{output_format}"
                dst = out_dir / filename

                # Check if file exists and handle overwrite behavior
                if dst.exists() and not overwrite:
                    print(f"[VideoShotSplitterV2] Skipping {dst.name} (file exists, overwrite=False)")
                    # Still add to the list if the file exists and is valid
                    if dst.stat().st_size > 0:
                        shot_videos.append(VideoFromFile(str(dst)))
                    continue

                subprocess.check_call(encode_cmd(output_format, work_path,
                                                 start.get_timecode(), n,
                                                 dst, reencode=reencode, overwrite=overwrite))
                shot_videos.append(VideoFromFile(str(dst)))
                print(f"[VideoShotSplitterV2] → {dst.name}")

            # clean temp stuff
            if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
            for p in self._temp_sources:
                try: os.remove(p)
                except: pass

            print(f"[VideoShotSplitterV2] ✓ Successfully created {len(shot_videos)} shots")
            return (shot_videos,)

        except Exception as e:
            print(f"[VideoShotSplitterV2-ERROR] {e}")
            return ([],)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _fixed_length_scenes(self, path: Path, secs: float):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for fixed-length splitting.")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = int(secs * fps)
        if not SCENEDETECT_AVAILABLE:
            from collections import namedtuple
            FrameTimecodeLocal = namedtuple("FT", "frame fps")
            FrameTimecodeLocal.get_frames   = lambda self: self.frame
            FrameTimecodeLocal.get_timecode = lambda self: f"{self.frame / self.fps:.3f}"
            FTC = FrameTimecodeLocal
        else:
            FTC = FrameTimecode
        scenes = [(FTC(f0, fps), FTC(min(f0 + step, total), fps))
                  for f0 in range(0, total, step)]
        cap.release()
        return scenes

    def _extract_video_path(self, vid):
        if isinstance(vid, str) and os.path.exists(vid):
            return vid
        if hasattr(vid, "_VideoFromFile__file"):
            p = getattr(vid, "_VideoFromFile__file")
            if isinstance(p, str) and os.path.exists(p):
                return p
        for attr in ("video_path", "filename", "path", "_path"):
            if hasattr(vid, attr):
                p = getattr(vid, attr)
                if isinstance(p, str) and os.path.exists(p):
                    return p
        if hasattr(vid, "save_to"):
            tmp = Path(tempfile.mkdtemp(prefix="splitshots_src_")) / "in.mp4"
            vid.save_to(str(tmp))
            self._temp_sources.append(tmp)
            return str(tmp)
        if isinstance(vid, dict):
            for k in ("video_path", "filename", "path"):
                if k in vid and os.path.exists(vid[k]):
                    return vid[k]
        s = str(vid)
        return s if os.path.exists(s) else None
