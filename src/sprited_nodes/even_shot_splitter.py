#!/usr/bin/env python3
"""
VideoEvenShotSplitter ───────────────────────────────────────────────────────────
Split an input video (from ComfyUI “Load Video”) into a specified number of evenly-sized shots.
Returns a list of VIDEO objects, ready for further processing.
"""

from __future__ import annotations
import os, tempfile
from pathlib import Path
from typing import List
from inspect import cleandoc
from datetime import datetime
import numpy as np

from comfy_api.input.video_types import VideoInput
from comfy_api.input_impl import VideoFromFile

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio.v3 as iio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

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
        if not IMAGEIO_AVAILABLE:
            raise RuntimeError("imageio is required for video encoding.")
            
        src_path = self._extract_video_path(video)
        if not src_path:
            raise RuntimeError("Unable to resolve video file path.")
        src_path = Path(src_path)
        out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_even_shots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video metadata
        cap = cv2.VideoCapture(str(src_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        logger.info(f"Splitting {total} frames into {num_shots} shots (fps={fps})")
        if num_shots < 1:
            num_shots = 1
        if num_shots > total:
            num_shots = total
        frames_per_shot = total // num_shots
        leftover = total % num_shots
        logger.info(f"frames_per_shot={frames_per_shot}, leftover={leftover}")
        
        # Read all frames from source video using imageio
        logger.info(f"Reading video frames from {src_path}")
        try:
            frames = iio.imread(str(src_path), plugin="FFMPEG")
        except Exception as e:
            logger.warning(f"Failed to read with FFMPEG plugin: {e}, trying default")
            frames = iio.imread(str(src_path))
        
        if frames.ndim == 3:  # Single frame
            frames = frames[None, ...]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if unique_filenames else ""
        shot_videos = []
        start_frame = 0
        
        for idx in range(num_shots):
            n_frames = frames_per_shot + (1 if idx < leftover else 0)
            logger.info(f"Chunk {idx}: start_frame={start_frame}, n_frames={n_frames}")
            assert n_frames > 0, f"Chunk {idx} has non-positive n_frames: {n_frames}"
            assert start_frame < total, f"Chunk {idx} start_frame {start_frame} >= total {total}"
            
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
            
            # Extract frames for this shot
            end_frame = start_frame + n_frames
            shot_frames = frames[start_frame:end_frame]
            
            logger.info(f"Chunk {idx}: writing {n_frames} frames to {dst}")
            
            # Write using imageio
            if output_format == "webp":
                iio.imwrite(str(dst), shot_frames, fps=fps, 
                           plugin="FFMPEG", codec="libwebp", 
                           output_params=["-lossless", "1", "-loop", "0"])
            elif output_format == "mp4":
                iio.imwrite(str(dst), shot_frames, fps=fps,
                           plugin="FFMPEG", codec="libx264",
                           output_params=["-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"])
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            assert dst.exists() and dst.stat().st_size > 0, f"Chunk {idx}: Output file {dst} was not created or is empty!"
            shot_videos.append(VideoFromFile(str(dst)))
            start_frame += n_frames
        
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
