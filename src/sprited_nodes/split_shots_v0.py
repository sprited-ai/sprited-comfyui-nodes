#!/usr/bin/env python3
"""
VideoShotSplitterV0 ─────────────────────────────────────────────────────────
Split an input video (from ComfyUI "Load Video") into exactly k equal-duration
shots. This is the simplest version that just divides the video evenly.

No scene detection, no adaptive thresholds - just pure equal-duration splitting.
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
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── helpers -----------------------------------------------------------------
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

# ── simple timecode class ---------------------------------------------------
class SimpleTimecode:
    def __init__(self, frame: int, fps: float):
        self.frame = frame
        self.fps = fps

    def get_frames(self) -> int:
        return self.frame

    def get_timecode(self) -> str:
        return f"{self.frame / self.fps:.3f}"

# ── node --------------------------------------------------------------------
class VideoShotSplitterV0:
    def __init__(self):
        self._temp_sources: list[Path] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "target_shot_count": ("INT", {"default": 5, "min": 1, "max": 100}),
                "output_format": (["mp4", "webp"], {"default": "mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "unique_filenames": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)          # list[VideoFromFile]
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("shot_videos",)
    FUNCTION      = "split"
    CATEGORY      = "Video/Edit/Simple"
    DESCRIPTION   = cleandoc(__doc__)

    # ── main ----------------------------------------------------------------
    def split(self, video, target_shot_count, output_format, reencode,
              output_dir="", overwrite=True, unique_filenames=True):

        try:
            src_path = self._extract_video_path(video)
            if not src_path:
                raise RuntimeError("Unable to resolve video file path.")
            src_path = Path(src_path)

            out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_shots_v0"
            out_dir.mkdir(parents=True, exist_ok=True)

            # WebP → temp MP4
            if src_path.suffix.lower() == ".webp":
                work_path = webp_to_tmp_mp4(src_path)
                tmp_dir   = work_path.parent
            else:
                work_path, tmp_dir = src_path, None

            # Create equal-duration scenes
            scenes = self._create_equal_duration_scenes(work_path, target_shot_count)
            print(f"[VideoShotSplitterV0] Creating {len(scenes)} equal-duration shots")

            # slice & wrap
            shot_videos = []
            # Generate timestamp for unique filenames if enabled
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if unique_filenames else ""

            for idx, (start, end) in enumerate(scenes):
                n = end.get_frames() - start.get_frames()
                if n <= 0:
                    n = 1  # Ensure at least 1 frame per shot

                # Create filename with optional timestamp for uniqueness
                if unique_filenames:
                    filename = f"shot-{timestamp}-{idx:03}.{output_format}"
                else:
                    filename = f"shot-{idx:03}.{output_format}"
                dst = out_dir / filename

                # Check if file exists and handle overwrite behavior
                if dst.exists() and not overwrite:
                    print(f"[VideoShotSplitterV0] Skipping {dst.name} (file exists, overwrite=False)")
                    # Still add to the list if the file exists and is valid
                    if dst.stat().st_size > 0:
                        shot_videos.append(VideoFromFile(str(dst)))
                    continue

                subprocess.check_call(encode_cmd(output_format, work_path,
                                                 start.get_timecode(), n,
                                                 dst, reencode=reencode, overwrite=overwrite))
                shot_videos.append(VideoFromFile(str(dst)))  # ← proper ComfyUI video type
                print(f"[VideoShotSplitterV0] → {dst.name}")

            # clean temp stuff
            if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
            for p in self._temp_sources:
                try: os.remove(p)
                except: pass

            print(f"[VideoShotSplitterV0] Successfully created {len(shot_videos)} shots")
            return (shot_videos,)

        except Exception as e:
            print(f"[VideoShotSplitterV0-ERROR] {e}")
            return ([],)

    # ── helpers -------------------------------------------------------------
    def _create_equal_duration_scenes(self, path: Path, target_k: int):
        """Create exactly k equal-duration scenes"""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for video analysis.")

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"[VideoShotSplitterV0] Video info: {total_frames} frames at {fps:.2f} fps")

        if total_frames < target_k:
            # Video too short, create single-frame shots and pad
            print(f"[VideoShotSplitterV0] Video too short ({total_frames} frames < {target_k} shots), using single frames")
            frames_per_shot = 1
        else:
            frames_per_shot = total_frames // target_k
            print(f"[VideoShotSplitterV0] Using {frames_per_shot} frames per shot")

        scenes = []
        for i in range(target_k):
            start_frame = i * frames_per_shot
            end_frame = min((i + 1) * frames_per_shot, total_frames)

            # Ensure last shot gets remaining frames
            if i == target_k - 1:
                end_frame = total_frames

            # Handle case where video is shorter than target_k
            if start_frame >= total_frames:
                # Use the last frame for remaining shots
                start_frame = total_frames - 1
                end_frame = total_frames

            scenes.append((SimpleTimecode(start_frame, fps), SimpleTimecode(end_frame, fps)))

        return scenes

    def _extract_video_path(self, vid):
        """Extract file path from various video object types"""
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
