#!/usr/bin/env python3
"""
VideoShotSplitter ───────────────────────────────────────────────────────────
ComfyUI node that splits an input video into individual shots.

Features
--------
• Accepts the VIDEO output of the built-in “Load Video” node.
• Automatic scene-cut detection (PySceneDetect) or fixed-length splitting.
• Handles MP4 or animated WebP in; outputs MP4 or loss-less WebP.
• Uses FFmpeg for the actual slicing (stream-copy when possible).

Install missing deps:
    pip install scenedetect opencv-python

Place this file in `ComfyUI/custom_nodes/`, then restart ComfyUI.

Author: you 😉
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile, glob
from pathlib import Path
from typing import List, Tuple
from inspect import cleandoc

# ── Optional deps -----------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_scenes(
    video: Path,
    *,
    detector_cls,
    threshold: float,
    min_len: int
) -> List[Tuple]:
    """Return [(start, end_exclusive), …] as FrameTimecode pairs."""
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
    """
    Convert animated WebP to a temporary loss-less MP4.
    Requires FFmpeg with WebP support OR the webpmux utility.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="splitshots_"))
    dst = tmp_dir / f"{src.stem}.mp4"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst),
    ]
    if subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size:
        return dst

    # Fallback: webpmux → PNGs → MP4
    if not shutil.which("webpmux"):
        raise RuntimeError(
            "Need 'webpmux' or an FFmpeg build with WebP support."
        )
    frames = tmp_dir / "frames"; frames.mkdir()
    subprocess.check_call([
        "webpmux", "-dump", str(src), "-o", str(frames / "frame.png")
    ])
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", "30", "-pattern_type", "glob",
        "-i", str(frames / "frame*.png"),
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", str(dst),
    ])
    for p in glob.glob(str(frames / "frame*.png")):
        os.remove(p)
    return dst

def encode_cmd(
    fmt: str,
    src: Path,
    start_ts: str,
    n_frames: int,
    dst: Path,
    *,
    reencode: bool
) -> List[str]:
    """Generate FFmpeg command that extracts `n_frames` starting at `start_ts`."""
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
        "-frames:v", str(n_frames),
    ]
    if fmt == "mp4":   # loss-less H.264
        common += ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
                   "-pix_fmt", "yuv444p"]
    elif fmt == "webp":
        common += ["-an", "-vcodec", "libwebp", "-lossless", "1",
                   "-preset", "default", "-loop", "0", "-vsync", "0"]
    else:
        raise ValueError(fmt)

    return common + [str(dst)]

# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class VideoShotSplitter:
    """
    Split a video (from Load Video) into per-shot clips.
    """

    # ── ComfyUI I/O spec ----------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "detector": (["content", "adaptive"], {
                    "default": "content",
                    "tooltip": "Scene-cut algorithm (ignored if seconds_per_shot > 0)"
                }),
                "threshold": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 50, "step": 0.1
                }),
                "min_scene_len": ("INT", {
                    "default": 15, "min": 1, "max": 300, "step": 1
                }),
                "output_format": (["mp4", "webp"], { "default": "mp4" }),
                "reencode": ("BOOLEAN", { "default": False }),
            },
            "optional": {
                "seconds_per_shot": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                    "tooltip": "Fixed-length chunking (0 = auto scene detect)"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Leave blank → <video_stem>_shots"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_file_paths",)
    FUNCTION = "split_video_shots"
    OUTPUT_NODE = False
    CATEGORY = "Video/Edit"
    DESCRIPTION = cleandoc(__doc__)

    # ── main function -------------------------------------------------------
    def split_video_shots(
        self,
        video,
        detector,
        threshold,
        min_scene_len,
        output_format,
        reencode,
        seconds_per_shot=0.0,
        output_dir="",
    ):
        try:
            # 1) Locate source file ------------------------------------------------
            src_path = self._extract_video_path(video)
            if not src_path:
                raise RuntimeError("Unable to resolve video file path.")
            src_path = Path(src_path)
            if not src_path.exists():
                raise FileNotFoundError(src_path)

            # 2) Prepare output folder --------------------------------------------
            out_dir = (Path(output_dir) if output_dir
                       else src_path.parent / f"{src_path.stem}_shots")
            out_dir.mkdir(parents=True, exist_ok=True)

            # 3) Handle animated WebP ---------------------------------------------
            work_path, tmp_dir = (
                (webp_to_tmp_mp4(src_path), tempfile.mkdtemp(prefix="splitshots_"))
                if src_path.suffix.lower() == ".webp"
                else (src_path, None)
            )

            # 4) Determine scenes --------------------------------------------------
            if seconds_per_shot > 0:
                if not CV2_AVAILABLE:
                    raise RuntimeError("OpenCV required for fixed-length splitting.")
                scenes = self._fixed_length_scenes(work_path, seconds_per_shot)
            else:
                if not SCENEDETECT_AVAILABLE:
                    raise RuntimeError("PySceneDetect required for auto scene detection.")
                Det = ContentDetector if detector == "content" else AdaptiveDetector
                scenes = detect_scenes(
                    work_path,
                    detector_cls=Det,
                    threshold=threshold,
                    min_len=min_scene_len
                )

            if not scenes:
                print("[VideoShotSplitter] No cuts detected.")
                return ("",)

            # 5) Slice video -------------------------------------------------------
            shot_paths = []
            for idx, (start, end) in enumerate(scenes):
                n_frames = end.get_frames() - start.get_frames()
                if n_frames <= 0:
                    continue
                dst = out_dir / f"shot-{idx:03}.{output_format}"
                cmd = encode_cmd(
                    output_format, work_path,
                    start.get_timecode(), n_frames,
                    dst, reencode=reencode
                )
                subprocess.check_call(cmd)
                shot_paths.append(str(dst))
                print(f"[VideoShotSplitter]  -> {dst.name}")

            # 6) Cleanup -----------------------------------------------------------
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return (",".join(shot_paths),)

        except Exception as e:
            print(f"[VideoShotSplitter-ERROR] {e}")
            return ("",)

    # ── helpers -------------------------------------------------------------
    def _fixed_length_scenes(self, video_path: Path, seconds: float):
        """Return scenes list for fixed-length splitting."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = int(seconds * fps)
        scenes = [(FrameTimecode(f0, fps),
                   FrameTimecode(min(f0 + step, total), fps))
                  for f0 in range(0, total, step)]
        cap.release()
        return scenes

    def _extract_video_path(self, vid):
        # print(f"[Yo] {dir(vid)}")
        """
        Supports:
          • raw string paths
          • comfy_api.VideoFromFile objects (video_path/filename attr)
          • dicts with a 'path'/'filename' key
        """
        if isinstance(vid, str):
            return vid

        for attr in ("video_path", "filename", "path", "_path"):
            if hasattr(vid, attr):
                p = getattr(vid, attr)
                if isinstance(p, str) and os.path.exists(p):
                    return p

        if isinstance(vid, dict):
            for k in ("video_path", "filename", "path"):
                if k in vid and os.path.exists(vid[k]):
                    return vid[k]

        s = str(vid)
        return s if os.path.exists(s) else None
