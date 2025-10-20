#!/usr/bin/env python3
"""
VideoEvenShotSplitter ───────────────────────────────────────────────────────────
Split an input video (from ComfyUI “Load Video”) into a specified number of evenly-sized shots.
Returns a list of VIDEO objects, ready for further processing.
"""

from __future__ import annotations
import os, shutil, subprocess, tempfile
from pathlib import Path
from typing import List
from inspect import cleandoc
from datetime import datetime

from comfy_api.input.video_types import VideoInput
from comfy_api.input_impl import VideoFromFile

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def encode_cmd(fmt: str, src: Path, start_ts: str, n_frames: int | None,
               dst: Path, *, reencode: bool, overwrite: bool = True) -> List[str]:
    base_args = ["-hide_banner", "-loglevel", "error"]
    if overwrite:
        base_args.append("-y")
    cmd = ["ffmpeg"] + base_args + ["-ss", start_ts, "-i", str(src)]
    if n_frames is not None and n_frames > 0:
        cmd += ["-frames:v", str(n_frames)]
    if fmt == "mp4" and not reencode:
        cmd += ["-c", "copy"]
    else:
        if fmt == "mp4":
            cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"]
        elif fmt == "webp":
            cmd += ["-an", "-vcodec", "libwebp", "-lossless", "1", "-preset", "default", "-loop", "0", "-vsync", "0"]
        else:
            raise ValueError(fmt)
    return cmd + [str(dst)]

class VideoEvenShotSplitter:
    def __init__(self):
        self._temp_sources: list[Path] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "num_shots": ("INT", {"default": 2, "min": 1, "max": 100}),
                "output_format": (["mp4", "webp"], {"default": "mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "unique_filenames": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("shot_videos",)
    FUNCTION      = "split_evenly"
    CATEGORY      = "Video/Edit"
    DESCRIPTION   = cleandoc(__doc__)

    def split_evenly(self, video, num_shots, output_format, reencode,
                     output_dir="", overwrite=True, unique_filenames=True):
        import logging
        logger = logging.getLogger("even_shot_splitter")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[even_shot_splitter] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for even splitting.")
        src_path = self._extract_video_path(video)
        if not src_path:
            raise RuntimeError("Unable to resolve video file path.")
        src_path = Path(src_path)
        out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_even_shots"
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Splitting {total} frames into {num_shots} shots (fps={fps})")
        if num_shots < 1:
            num_shots = 1
        if num_shots > total:
            num_shots = total
        frames_per_shot = total // num_shots
        leftover = total % num_shots
        logger.info(f"frames_per_shot={frames_per_shot}, leftover={leftover}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if unique_filenames else ""
        shot_videos = []
        start_frame = 0
        for idx in range(num_shots):
            n_frames = frames_per_shot + (1 if idx < leftover else 0)
            logger.info(f"Chunk {idx}: start_frame={start_frame}, n_frames={n_frames}")
            assert n_frames > 0, f"Chunk {idx} has non-positive n_frames: {n_frames}"
            assert start_frame < total, f"Chunk {idx} start_frame {start_frame} >= total {total}"
            start_sec = start_frame / fps
            if unique_filenames:
                filename = f"even-shot-{timestamp}-{idx:03}.{output_format}"
            else:
                filename = f"even-shot-{idx:03}.{output_format}"
            dst = out_dir / filename
            if dst.exists() and not overwrite:
                if dst.stat().st_size > 0:
                    logger.info(f"Chunk {idx}: using existing file {dst}")
                    shot_videos.append(VideoFromFile(str(dst)))
                else:
                    logger.warning(f"Chunk {idx}: existing file {dst} is empty!")
                start_frame += n_frames
                continue
            logger.info(f"Chunk {idx}: ffmpeg -ss {start_sec:.3f} -frames:v {n_frames} ... -> {dst}")
            subprocess.check_call(
                encode_cmd(output_format, src_path, f"{start_sec:.3f}", n_frames,
                           dst, reencode=reencode, overwrite=overwrite)
            )
            assert dst.exists() and dst.stat().st_size > 0, f"Chunk {idx}: Output file {dst} was not created or is empty!"
            shot_videos.append(VideoFromFile(str(dst)))
            start_frame += n_frames
        cap.release()
        logger.info(f"Total output chunks: {len(shot_videos)}")
        assert len(shot_videos) == num_shots, f"Expected {num_shots} output videos, got {len(shot_videos)}"
        for p in self._temp_sources:
            try: os.remove(p)
            except: pass
        return (shot_videos,)

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
            tmp = Path(tempfile.mkdtemp(prefix="evenshots_src_")) / "in.mp4"
            vid.save_to(str(tmp))
            self._temp_sources.append(tmp)
            return str(tmp)
        if isinstance(vid, dict):
            for k in ("video_path", "filename", "path"):
                if k in vid and os.path.exists(vid[k]):
                    return vid[k]
        s = str(vid)
        return s if os.path.exists(s) else None
