#!/usr/bin/env python3
"""
split_shots.py — Split an input clip into one file per shot using
PySceneDetect + FFmpeg.  Supports MP4 or animated WebP in, MP4/WebP out.
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple

# ── PySceneDetect -----------------------------------------------------------
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode

# ---------------------------------------------------------------------------
# Scene detection helpers
# ---------------------------------------------------------------------------

def detect_scenes(
    video: Path, *, detector_cls, threshold: float, min_len: int
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Return [(start, end_exclusive), …] in FrameTimecodes."""
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
