#!/usr/bin/env python3
"""
VideoShotSplitter ───────────────────────────────────────────────────────────
Split an input video (from ComfyUI “Load Video”) into individual shots.

• Automatic scene detection (PySceneDetect) or fixed-length splitting.
• Handles MP4 + animated WebP in, MP4/WebP out (loss-less when requested).
• Uses stream-copy for MP4 unless “reencode” is True.

Install deps you don’t already have:
    pip install scenedetect opencv-python
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile, glob
from pathlib import Path
from typing import List, Tuple
from inspect import cleandoc

# ── optional deps ──────────────────────────────────────────────────────────
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

# ── helper funcs ───────────────────────────────────────────────────────────
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
    # fallback: webpmux → PNG → MP4
    if not shutil.which("webpmux"):
        raise RuntimeError("Need 'webpmux' or FFmpeg compiled with WebP.")
    frames = tmp_dir / "frames"; frames.mkdir()
    subprocess.check_call(["webpmux", "-dump", str(src), "-o", str(frames / "frm.png")])
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", "30", "-pattern_type", "glob",
        "-i", str(frames / "frm*.png"),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst)
    ])
    for p in glob.glob(str(frames / "frm*.png")): os.remove(p)
    return dst

def encode_cmd(fmt: str, src: Path, start_ts: str, n_frames: int,
               dst: Path, *, reencode: bool) -> List[str]:
    if fmt == "mp4" and not reencode:
        return ["ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", start_ts, "-i", str(src),
                "-frames:v", str(n_frames), "-c", "copy", str(dst)]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
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

# ── node ───────────────────────────────────────────────────────────────────
class VideoShotSplitter:
    def __init__(self):
        self._temp_sources: list[Path] = []

    # ── ComfyUI spec ───────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "detector": (["content", "adaptive"], {"default": "content"}),
                "threshold": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50}),
                "min_scene_len": ("INT", {"default": 15, "min": 1, "max": 300}),
                "output_format": (["mp4", "webp"], {"default": "mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seconds_per_shot": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0,
                    "tooltip": "0 = use scene detection"
                }),
                "output_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_file_paths",)
    FUNCTION      = "split"
    OUTPUT_NODE   = False
    CATEGORY      = "Video/Edit"
    DESCRIPTION   = cleandoc(__doc__)

    # ── main ---------------------------------------------------------------
    def split(self, video, detector, threshold, min_scene_len,
              output_format, reencode,
              seconds_per_shot=0.0, output_dir=""):

        try:
            src_path = self._extract_video_path(video)
            if not src_path:
                raise RuntimeError("Unable to resolve video file path.")
            src_path = Path(src_path)

            out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_shots"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Animated WebP handling
            if src_path.suffix.lower() == ".webp":
                work_path = webp_to_tmp_mp4(src_path)
                tmp_dir   = work_path.parent  # real temp dir
            else:
                work_path, tmp_dir = src_path, None

            # scene list
            if seconds_per_shot > 0:
                scenes = self._fixed_length_scenes(work_path, seconds_per_shot)
            else:
                if not SCENEDETECT_AVAILABLE:
                    raise RuntimeError("PySceneDetect not installed.")
                Det = ContentDetector if detector == "content" else AdaptiveDetector
                scenes = detect_scenes(work_path, detector_cls=Det,
                                       threshold=threshold, min_len=min_scene_len)

            if not scenes:
                print("[VideoShotSplitter] No cuts detected.")
                return ("",)

            # slice
            shots = []
            for idx, (start, end) in enumerate(scenes):
                n = end.get_frames() - start.get_frames()
                if n <= 0: continue
                dst = out_dir / f"shot-{idx:03}.{output_format}"
                subprocess.check_call(encode_cmd(output_format, work_path,
                                                 start.get_timecode(), n,
                                                 dst, reencode=reencode))
                shots.append(str(dst))
                print(f"[VideoShotSplitter] → {dst.name}")

            # tidy up
            if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
            for p in self._temp_sources:
                try: os.remove(p)
                except: pass

            return (",".join(shots),)

        except Exception as e:
            print(f"[VideoShotSplitter-ERROR] {e}")
            return ("",)

    # ── helpers ------------------------------------------------------------
    def _fixed_length_scenes(self, path: Path, secs: float):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for fixed-length splitting.")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = int(secs * fps)

        # fallback FrameTimecode when scenedetect missing
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
        # 1) raw str
        if isinstance(vid, str) and os.path.exists(vid):
            return vid

        # 2) private attr used by comfy_api.VideoFromFile
        if hasattr(vid, "_VideoFromFile__file"):
            p = getattr(vid, "_VideoFromFile__file")
            if isinstance(p, str) and os.path.exists(p):
                return p

        # 3) public-ish attrs
        for attr in ("video_path", "filename", "path", "_path"):
            if hasattr(vid, attr):
                p = getattr(vid, attr)
                if isinstance(p, str) and os.path.exists(p):
                    return p

        # 4) object provides save_to() -> write temp file
        if hasattr(vid, "save_to"):
            tmp = Path(tempfile.mkdtemp(prefix="splitshots_src_")) / "in.mp4"
            vid.save_to(str(tmp))
            self._temp_sources.append(tmp)
            return str(tmp)

        # 5) dict hand-off
        if isinstance(vid, dict):
            for k in ("video_path", "filename", "path"):
                if k in vid and os.path.exists(vid[k]):
                    return vid[k]

        # 6) last resort
        s = str(vid)
        return s if os.path.exists(s) else None
