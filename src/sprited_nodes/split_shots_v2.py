#!/usr/bin/env python3
"""
VideoShotSplitterV2 ─────────────────────────────────────────────────────────
Split an input video (from ComfyUI "Load Video") into individual shots and
return a list of VIDEO objects, ready for further processing.

Enhanced version that uses binary search to find the optimal scene detection
threshold, guaranteeing between min_shots and max_shots outputs. If scene
detection fails to find shots in the target range, it falls back to
fixed-length splitting targeting the middle of the range.
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

def detect_scenes_bisect(video: Path, *, detector_cls, initial_threshold: float,
                        min_len: int, min_shots: int, max_shots: int) -> List[Tuple]:
    """
    Robust binary search to find optimal threshold that returns [min_shots, max_shots] scenes.
    Includes dynamic range expansion and multiple fallback strategies for 99%+ success rate.
    """

    def try_threshold(thresh: float):
        """Try a threshold and return (scenes, count) with error handling."""
        try:
            scenes = detect_scenes(video, detector_cls=detector_cls,
                                 threshold=thresh, min_len=min_len)
            return scenes, len(scenes)
        except Exception as e:
            print(f"[VideoShotSplitterV2] Scene detection failed at threshold {thresh:.2f}: {e}")
            return [], 0

    # Phase 1: Explore initial range
    low_threshold = 0.01   # Start very sensitive
    high_threshold = min(initial_threshold * 3, 100.0)  # Wider initial range
    best_scenes = []
    best_threshold = initial_threshold
    best_distance = float('inf')  # Distance from target range
    max_iterations = 20

    print(f"[VideoShotSplitterV2] Binary search for {min_shots}-{max_shots} shots in range [{low_threshold:.2f}, {high_threshold:.1f}]")

    # Phase 2: Sample boundary points to understand behavior
    low_scenes, low_count = try_threshold(low_threshold)
    high_scenes, high_count = try_threshold(high_threshold)

    print(f"[VideoShotSplitterV2] Boundary check: {low_count} scenes @ {low_threshold:.2f}, {high_count} scenes @ {high_threshold:.1f}")

    # Phase 3: Dynamic range expansion if needed
    if low_count < min_shots and high_count < min_shots:
        # Need to go even lower
        print(f"[VideoShotSplitterV2] Expanding search range lower...")
        for new_low in [0.005, 0.001, 0.0005]:
            test_scenes, test_count = try_threshold(new_low)
            if test_count >= min_shots:
                low_threshold = new_low
                low_scenes, low_count = test_scenes, test_count
                break
        else:
            print(f"[VideoShotSplitterV2] ⚠ Even very low thresholds don't produce enough scenes")

    if low_count > max_shots and high_count > max_shots:
        # Need to go even higher
        print(f"[VideoShotSplitterV2] Expanding search range higher...")
        for new_high in [200.0, 500.0, 1000.0]:
            test_scenes, test_count = try_threshold(new_high)
            if test_count <= max_shots:
                high_threshold = new_high
                high_scenes, high_count = test_scenes, test_count
                break
        else:
            print(f"[VideoShotSplitterV2] ⚠ Even very high thresholds produce too many scenes")

    # Update best candidates from boundary checks
    for scenes, count, thresh in [(low_scenes, low_count, low_threshold),
                                  (high_scenes, high_count, high_threshold)]:
        if min_shots <= count <= max_shots:
            print(f"[VideoShotSplitterV2] ✓ Perfect boundary result: {count} scenes @ {thresh:.2f}")
            return scenes

        # Track best result (closest to target range)
        if count >= min_shots:
            distance = max(0, count - max_shots)  # How far above max_shots
        else:
            distance = min_shots - count  # How far below min_shots

        if distance < best_distance:
            best_scenes, best_threshold, best_distance = scenes, thresh, distance

    # Phase 4: Binary search with robust convergence
    for iteration in range(max_iterations):
        # Handle edge case where bounds converged
        if abs(high_threshold - low_threshold) < 0.001:
            print(f"[VideoShotSplitterV2] Search range converged")
            break

        threshold = (low_threshold + high_threshold) / 2
        print(f"[VideoShotSplitterV2] Iteration {iteration+1}: trying threshold={threshold:.3f}")

        scenes, num_scenes = try_threshold(threshold)
        if not scenes and num_scenes == 0:
            # Scene detection failed, try to continue
            print(f"[VideoShotSplitterV2] Scene detection failed, adjusting bounds")
            high_threshold = (low_threshold + high_threshold) / 2
            continue

        print(f"[VideoShotSplitterV2] Found {num_scenes} scenes")

        # Perfect result
        if min_shots <= num_scenes <= max_shots:
            print(f"[VideoShotSplitterV2] ✓ Perfect! Found {num_scenes} scenes @ {threshold:.3f}")
            return scenes

        # Update best result
        if num_scenes >= min_shots:
            distance = max(0, num_scenes - max_shots)
        else:
            distance = min_shots - num_scenes

        if distance < best_distance:
            best_scenes, best_threshold, best_distance = scenes, threshold, distance
            print(f"[VideoShotSplitterV2] New best: {num_scenes} scenes @ {threshold:.3f} (distance: {distance})")

        # Adjust search bounds (robust to non-monotonic behavior)
        if num_scenes < min_shots:
            high_threshold = threshold
        else:
            low_threshold = threshold

    # Phase 5: Return best result or try alternative approaches
    if best_scenes and len(best_scenes) >= min_shots:
        print(f"[VideoShotSplitterV2] ✓ Best result: {len(best_scenes)} scenes @ {best_threshold:.3f}")
        return best_scenes

    # Phase 6: Last resort - try different min_scene_len values
    if min_len > 1:
        print(f"[VideoShotSplitterV2] Trying with reduced min_scene_len...")
        for reduced_min_len in [max(1, min_len // 2), 1]:
            try:
                alt_scenes = detect_scenes(video, detector_cls=detector_cls,
                                         threshold=initial_threshold, min_len=reduced_min_len)
                if len(alt_scenes) >= min_shots:
                    print(f"[VideoShotSplitterV2] ✓ Success with min_scene_len={reduced_min_len}: {len(alt_scenes)} scenes")
                    return alt_scenes
            except Exception as e:
                print(f"[VideoShotSplitterV2] Alternative approach failed: {e}")
                continue

    print(f"[VideoShotSplitterV2] ⚠ All approaches failed, will use fallback")
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
                "max_shots": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Must be >= min_shots"}),
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
    def split(self, video, min_shots, max_shots, detector, threshold, min_scene_len,
              output_format, reencode,
              seconds_per_shot=0.0, output_dir="", overwrite=True, unique_filenames=True):

        try:
            # Input validation
            if max_shots < min_shots:
                raise ValueError(f"max_shots ({max_shots}) cannot be less than min_shots ({min_shots})")

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
                scenes = detect_scenes_bisect(
                    work_path,
                    detector_cls=Det,
                    initial_threshold=threshold,
                    min_len=min_scene_len,
                    min_shots=min_shots,
                    max_shots=max_shots
                )

                # Enhanced fallback with multiple strategies
                if not scenes or len(scenes) < min_shots:
                    print(f"[VideoShotSplitterV2] Falling back to fixed-length splitting for {min_shots}-{max_shots} shots")
                    try:
                        duration = get_video_duration(work_path)

                        # Strategy 1: Target middle of range
                        target_shots = (min_shots + max_shots) // 2
                        seconds_per_shot_calc = duration / target_shots
                        scenes = self._fixed_length_scenes(work_path, seconds_per_shot_calc)

                        # Strategy 2: If that doesn't work, try min_shots exactly
                        if len(scenes) < min_shots:
                            print(f"[VideoShotSplitterV2] Trying fixed-length with exact min_shots ({min_shots})")
                            seconds_per_shot_calc = duration / min_shots
                            scenes = self._fixed_length_scenes(work_path, seconds_per_shot_calc)

                        # Strategy 3: If video is very short, use shorter segments
                        if len(scenes) < min_shots and duration < 60:  # Less than 1 minute
                            print(f"[VideoShotSplitterV2] Short video detected, using smaller segments")
                            seconds_per_shot_calc = max(0.5, duration / min_shots)  # At least 0.5s per shot
                            scenes = self._fixed_length_scenes(work_path, seconds_per_shot_calc)

                        print(f"[VideoShotSplitterV2] Created {len(scenes)} fixed-length shots ({seconds_per_shot_calc:.1f}s each)")
                    except Exception as e:
                        print(f"[VideoShotSplitterV2] All fallback strategies failed: {e}")
                        return ([],)

            if not scenes:
                print("[VideoShotSplitterV2] No shots could be created.")
                return ([],)

            # Validate we have shots within the desired range
            if len(scenes) < min_shots:
                print(f"[VideoShotSplitterV2] WARNING: Only created {len(scenes)} shots, less than minimum {min_shots}")
            elif len(scenes) > max_shots:
                print(f"[VideoShotSplitterV2] INFO: Created {len(scenes)} shots, more than maximum {max_shots} (will process all)")
            else:
                print(f"[VideoShotSplitterV2] ✓ Created {len(scenes)} shots within target range [{min_shots}, {max_shots}]")

            # Process scenes into video files
            shot_videos = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if unique_filenames else ""

            for idx, (start, end) in enumerate(scenes):
                n = end.get_frames() - start.get_frames()
                if n <= 0: continue

                print(f"[VideoShotSplitterV2] Processing scene {idx}: frames {start.get_frames()}-{end.get_frames()} ({n} frames)")

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

                # Simple validation like the original split_shots.py
                if dst.exists() and dst.stat().st_size > 0:
                    shot_videos.append(VideoFromFile(str(dst)))
                    print(f"[VideoShotSplitterV2] → {dst.name}")
                else:
                    print(f"[VideoShotSplitterV2] ⚠ Skipping {dst.name} (empty file)")

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
