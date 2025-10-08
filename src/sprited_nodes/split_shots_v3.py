#!/usr/bin/env python3
"""
VideoShotSplitterV3 ─────────────────────────────────────────────────────────
Split an input video (from ComfyUI "Load Video") into exactly k shots using
adaptive threshold tuning. This version guarantees exactly k output videos.

The splitter will automatically adjust the detection threshold to achieve the
target number of shots. If natural scene detection cannot produce k shots,
it will fall back to equal-duration splitting to ensure exactly k outputs.
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

# ── helpers (unchanged) -----------------------------------------------------
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

# ── node --------------------------------------------------------------------
class VideoShotSplitterV3:
    def __init__(self):
        self._temp_sources: list[Path] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "target_shot_count": ("INT", {"default": 5, "min": 1, "max": 100}),
                "detector": (["content", "adaptive"], {"default": "content"}),
                "threshold": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 50}),
                "min_scene_len": ("INT", {"default": 15, "min": 1, "max": 300}),
                "output_format": (["mp4", "webp"], {"default": "mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "unique_filenames": ("BOOLEAN", {"default": True}),
                "max_retries": ("INT", {"default": 15, "min": 5, "max": 30}),
            },
        }

    RETURN_TYPES = ("VIDEO",)          # list[VideoFromFile]
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("shot_videos",)
    FUNCTION      = "split"
    CATEGORY      = "Video/Edit/Advanced"
    DESCRIPTION   = cleandoc(__doc__)

    # ── main ----------------------------------------------------------------
    def split(self, video, target_shot_count, detector, threshold, min_scene_len,
              output_format, reencode,
              output_dir="", overwrite=True, unique_filenames=True, max_retries=15):

        try:
            src_path = self._extract_video_path(video)
            if not src_path:
                raise RuntimeError("Unable to resolve video file path.")
            src_path = Path(src_path)

            out_dir = Path(output_dir) if output_dir else src_path.parent / f"{src_path.stem}_shots_v3"
            out_dir.mkdir(parents=True, exist_ok=True)

            # WebP → temp MP4
            if src_path.suffix.lower() == ".webp":
                work_path = webp_to_tmp_mp4(src_path)
                tmp_dir   = work_path.parent
            else:
                work_path, tmp_dir = src_path, None

            # Find optimal threshold to get exactly k shots
            if not SCENEDETECT_AVAILABLE:
                print("[VideoShotSplitterV3] PySceneDetect not available, using equal-duration fallback")
                scenes = self._create_equal_duration_scenes(work_path, target_shot_count)
            else:
                Det = ContentDetector if detector == "content" else AdaptiveDetector
                optimal_threshold, scenes = self._find_optimal_threshold(
                    work_path, Det, target_shot_count, threshold, min_scene_len, max_retries
                )
                print(f"[VideoShotSplitterV3] Using threshold: {optimal_threshold:.2f} for {len(scenes)} shots")

            # Ensure we have exactly k shots (fallback if needed)
            if len(scenes) != target_shot_count:
                print(f"[VideoShotSplitterV3] Scene detection yielded {len(scenes)} shots, forcing {target_shot_count}")
                scenes = self._force_k_shots(work_path, scenes, target_shot_count)

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
                    print(f"[VideoShotSplitterV3] Skipping {dst.name} (file exists, overwrite=False)")
                    # Still add to the list if the file exists and is valid
                    if dst.stat().st_size > 0:
                        shot_videos.append(VideoFromFile(str(dst)))
                    continue

                subprocess.check_call(encode_cmd(output_format, work_path,
                                                 start.get_timecode(), n,
                                                 dst, reencode=reencode, overwrite=overwrite))
                shot_videos.append(VideoFromFile(str(dst)))  # ← proper ComfyUI video type
                print(f"[VideoShotSplitterV3] → {dst.name}")

            # clean temp stuff
            if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
            for p in self._temp_sources:
                try: os.remove(p)
                except: pass

            # Ensure we return exactly k shots
            if len(shot_videos) != target_shot_count:
                print(f"[VideoShotSplitterV3-WARNING] Expected {target_shot_count} shots, got {len(shot_videos)}")

            return (shot_videos,)

        except Exception as e:
            print(f"[VideoShotSplitterV3-ERROR] {e}")
            return ([],)

    # ── k-shots helpers ----------------------------------------------------
    def _find_optimal_threshold(self, path: Path, detector_cls, target_k: int,
                               initial_threshold: float, min_len: int, max_retries: int):
        """Binary search to find threshold that yields closest to k shots"""

        # Define search bounds
        min_threshold = 0.1
        max_threshold = 50.0
        best_threshold = initial_threshold
        best_scenes = []
        best_diff = float('inf')

        # Try initial threshold first
        try:
            scenes = detect_scenes(path, detector_cls=detector_cls,
                                 threshold=initial_threshold, min_len=min_len)
            diff = abs(len(scenes) - target_k)
            if diff < best_diff:
                best_diff = diff
                best_threshold = initial_threshold
                best_scenes = scenes

            if len(scenes) == target_k:
                return initial_threshold, scenes
        except:
            pass

        # Binary search
        low, high = min_threshold, max_threshold

        for attempt in range(max_retries):
            if best_diff == 0:  # Found exact match
                break

            # Try midpoint
            mid_threshold = (low + high) / 2.0

            try:
                scenes = detect_scenes(path, detector_cls=detector_cls,
                                     threshold=mid_threshold, min_len=min_len)
                scene_count = len(scenes)
                diff = abs(scene_count - target_k)

                # Update best if this is closer
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = mid_threshold
                    best_scenes = scenes

                # Adjust search bounds
                if scene_count < target_k:
                    # Too few scenes, need lower threshold (more sensitive)
                    high = mid_threshold
                elif scene_count > target_k:
                    # Too many scenes, need higher threshold (less sensitive)
                    low = mid_threshold
                else:
                    # Exact match!
                    return mid_threshold, scenes

            except Exception as e:
                print(f"[VideoShotSplitterV3] Threshold {mid_threshold:.2f} failed: {e}")
                # Adjust bounds to avoid this threshold
                if mid_threshold < initial_threshold:
                    low = mid_threshold
                else:
                    high = mid_threshold

        # If we couldn't find good scenes, fall back to equal duration
        if not best_scenes or best_diff > target_k // 2:
            print(f"[VideoShotSplitterV3] Scene detection failed, using equal-duration fallback")
            return best_threshold, self._create_equal_duration_scenes(path, target_k)

        return best_threshold, best_scenes

    def _create_equal_duration_scenes(self, path: Path, target_k: int):
        """Create exactly k equal-duration scenes"""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for equal-duration splitting.")

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames < target_k:
            # Video too short, create single-frame shots
            frames_per_shot = 1
        else:
            frames_per_shot = total_frames // target_k

        # Use FrameTimecode if available, otherwise create simple equivalent
        if SCENEDETECT_AVAILABLE:
            FTC = FrameTimecode
        else:
            from collections import namedtuple
            FrameTimecodeLocal = namedtuple("FT", "frame fps")
            FrameTimecodeLocal.get_frames = lambda self: self.frame
            FrameTimecodeLocal.get_timecode = lambda self: f"{self.frame / self.fps:.3f}"
            FTC = FrameTimecodeLocal

        scenes = []
        for i in range(target_k):
            start_frame = i * frames_per_shot
            end_frame = min((i + 1) * frames_per_shot, total_frames)

            # Ensure last shot gets remaining frames
            if i == target_k - 1:
                end_frame = total_frames

            scenes.append((FTC(start_frame, fps), FTC(end_frame, fps)))

        return scenes

    def _force_k_shots(self, path: Path, scenes: List, target_k: int):
        """Force exactly k shots by merging or splitting existing scenes"""
        current_count = len(scenes)

        if current_count == target_k:
            return scenes

        if current_count > target_k:
            # Too many scenes - merge adjacent scenes
            return self._merge_scenes(scenes, target_k)
        else:
            # Too few scenes - split longest scenes or pad with equal duration
            return self._split_or_pad_scenes(path, scenes, target_k)

    def _merge_scenes(self, scenes: List, target_k: int):
        """Merge adjacent scenes to reduce count to target_k"""
        if len(scenes) <= target_k:
            return scenes

        merged = scenes[:]

        while len(merged) > target_k:
            # Find shortest gap between adjacent scenes to merge
            min_gap = float('inf')
            merge_idx = 0

            for i in range(len(merged) - 1):
                gap = merged[i+1][0].get_frames() - merged[i][1].get_frames()
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i

            # Merge scenes at merge_idx and merge_idx+1
            start = merged[merge_idx][0]
            end = merged[merge_idx + 1][1]
            merged[merge_idx] = (start, end)
            merged.pop(merge_idx + 1)

        return merged

    def _split_or_pad_scenes(self, path: Path, scenes: List, target_k: int):
        """Split scenes or create additional equal-duration scenes to reach target_k"""
        current_count = len(scenes)
        needed = target_k - current_count

        # If we have some scenes, try to split the longest ones
        if scenes and needed <= len(scenes):
            return self._split_longest_scenes(scenes, needed)

        # Otherwise, fall back to equal duration for all k shots
        return self._create_equal_duration_scenes(path, target_k)

    def _split_longest_scenes(self, scenes: List, splits_needed: int):
        """Split the longest scenes to create additional shots"""
        scenes_copy = scenes[:]

        for _ in range(splits_needed):
            # Find longest scene
            longest_idx = 0
            longest_duration = 0

            for i, (start, end) in enumerate(scenes_copy):
                duration = end.get_frames() - start.get_frames()
                if duration > longest_duration:
                    longest_duration = duration
                    longest_idx = i

            # Split the longest scene in half
            start, end = scenes_copy[longest_idx]
            mid_frame = (start.get_frames() + end.get_frames()) // 2

            if SCENEDETECT_AVAILABLE:
                FTC = FrameTimecode
                fps = start.framerate
            else:
                from collections import namedtuple
                FrameTimecodeLocal = namedtuple("FT", "frame fps")
                FrameTimecodeLocal.get_frames = lambda self: self.frame
                FrameTimecodeLocal.get_timecode = lambda self: f"{self.frame / self.fps:.3f}"
                FTC = FrameTimecodeLocal
                fps = getattr(start, 'fps', 30)

            mid_tc = FTC(mid_frame, fps)

            # Replace original scene with two split scenes
            scenes_copy[longest_idx] = (start, mid_tc)
            scenes_copy.insert(longest_idx + 1, (mid_tc, end))

        return scenes_copy

    # ── helpers (unchanged) ------------------------------------------------
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
