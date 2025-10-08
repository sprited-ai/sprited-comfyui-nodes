# ────────────────────────────────────────────────────────────────────────────
# LoopTrimNode – detect best seamless loop and spit out the trimmed clip
# ────────────────────────────────────────────────────────────────────────────
import json, tempfile, math, os
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image
import imagehash

# ComfyUI video types
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    # Fallback if ComfyUI API is not available
    VideoFromFile = str

# ---- hashing helpers (unchanged) ------------------------------------------
def phash_vec(img):
    return imagehash.phash(img).hash.astype(np.uint8).flatten()

def frame_distance(a, b):
    return np.count_nonzero(a ^ b)

def frame_vec(img, size=16):
    return np.array(
        img.convert("L").resize((size, size), Image.NEAREST),
        dtype=np.float32
    ).flatten()

def period_autocorr(frames, min_len=30, max_len=120):
    X = np.stack([frame_vec(f) for f in frames], axis=0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    best_k, best_sim = 0, -1
    for k in range(min_len, min(max_len, len(X)//2) + 1):
        sim = (X[:-k] * X[k:]).sum(axis=1).mean()
        if sim > best_sim:
            best_sim, best_k = sim, k
    return best_k, best_sim

def motion_profile(frames):
    mp = []
    for i in range(len(frames) - 1):
        a = np.asarray(frames[i], dtype=np.float32)
        b = np.asarray(frames[i+1], dtype=np.float32)
        mp.append(((a - b) ** 2).mean())
    return np.array(mp)

def score_segment(hashes, motion, start, length, lam=0.5):
    end = start + length
    seam = frame_distance(hashes[start], hashes[end])
    avg_mot = motion[start:end].mean()
    total = seam + lam * (1.0 / (avg_mot + 1e-6))
    return total, seam, avg_mot

def detect_best_loop(hashes, frames, min_len=20, max_len=120, lam=0.5):
    period, _ = period_autocorr(frames, min_len, max_len)
    period = max(period, min_len)
    motion = motion_profile(frames)
    best = (math.inf, 0, 0, 0, 0)
    for start in range(0, len(frames) - period - 1):
        total, seam, avg_mot = score_segment(hashes, motion, start, period, lam)
        if total < best[0]:
            best = (total, start, start + period, seam, avg_mot)
    return best  # total, start, end, seam, avg_mot

def save_trimmed(frames, start, end, out_path, fps):
    sliced = frames[start:end+1]
    if out_path.lower().endswith(".gif"):
        sliced[0].save(
            out_path, save_all=True, append_images=sliced[1:],
            loop=0, duration=int(1000/fps), disposal=2
        )
    elif out_path.lower().endswith(".webp"):
        sliced[0].save(
            out_path, save_all=True, append_images=sliced[1:],
            loop=0, duration=int(1000/fps),
            method=6, lossless=False, quality=80
        )
    else:  # mp4
        arr = [np.array(f) for f in sliced]
        iio.imwrite(out_path, arr, fps=fps,
                    codec="libx264", quality=8, pixelformat="yuv420p")

# ---- main worker ----------------------------------------------------------
def _cut_loop_work(
    input_path: str,
    out_path: str,
    min_gap: int,
    max_gap: int,
    threshold: int,
    lam: float,
    limit: int = 0
):
    fps = 12.0
    # Load frames (FFmpeg for videos, PIL for webp/gif/apng)
    try:
        if input_path.lower().endswith('.webp'):
            img = Image.open(input_path)
            frames = []
            try:
                while True:
                    frames.append(img.copy())
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            if hasattr(img, 'info') and 'duration' in img.info:
                fps = 1000.0 / img.info['duration'] if img.info['duration'] else fps
        else:
            try:
                meta = iio.immeta(input_path, plugin="FFMPEG")
                fps = meta.get("fps", fps)
                frames_np = iio.imread(input_path, plugin="FFMPEG")
            except Exception:
                frames_np = iio.imread(input_path)
            if frames_np.ndim == 3:
                frames_np = frames_np[None, ...]
            frames = [Image.fromarray(f) for f in frames_np]
    except Exception as e:
        raise RuntimeError(f"Could not read {input_path}: {e}")

    if limit > 0:
        frames = frames[:limit]

    hashes = [phash_vec(f) for f in frames]
    total, start, end, seam, avg_mot = detect_best_loop(
        hashes, frames, min_len=min_gap, max_len=max_gap, lam=lam
    )

    info = {
        "start": start,
        "end": end,
        "length": end - start,
        "seam": seam,
        "avg_motion": round(float(avg_mot), 6),
        "score": round(float(total), 6),
        "fps": fps
    }

    # Always save the best loop found, regardless of seam quality
    save_trimmed(frames, start, end - 1, out_path, fps)
    info["saved"] = True

    return info, out_path

# ── ComfyUI node -----------------------------------------------------------
class LoopTrimNode:
    """
    Detects the best seamless loop inside an animated clip and outputs
    the trimmed VIDEO plus a JSON summary of what it found.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "min_gap":    ("INT", {"default": 10, "min": 2,  "max": 240}),
                "max_gap":    ("INT", {"default": 120,"min": 10, "max": 600}),
                "threshold":  ("INT", {"default": 2,  "min": 0,  "max": 64}),
                "lambda_":    ("FLOAT", {"default": 0.5,"min": 0.0, "max": 5.0, "step": 0.1}),
                "format":     (["mp4", "gif", "webp"], {"default": "webp"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
            }
        }

    # one VIDEO + one STRING (json)
    RETURN_TYPES     = ("VIDEO", "STRING")
    RETURN_NAMES     = ("trimmed_video", "info_json")
    CATEGORY         = "Video/Edit"
    FUNCTION         = "run"

    def run(self, video, min_gap, max_gap, threshold,
            lambda_, format, output_dir=""):

        # 1. Resolve path of input
        src = self._resolve_path(video)
        if not src:
            raise RuntimeError("Cannot resolve input video path.")

        # 2. Decide output path
        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="looptrim_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(src).stem}_loop.{format}"

        # 3. Do the work
        info, trimmed = _cut_loop_work(
            src, str(out_path),
            min_gap=min_gap, max_gap=max_gap,
            threshold=threshold, lam=lambda_
        )

        # Always return the result - no more failure on poor seam quality
        def to_builtin(o):
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        return (VideoFromFile(trimmed), json.dumps(info, indent=2, default=to_builtin))

    # helper to find the file path inside a VIDEO object
    def _resolve_path(self, vid):
        if isinstance(vid, str) and os.path.exists(vid):
            return vid
        for attr in ("video_path", "filename", "_VideoFromFile__file", "path"):
            if hasattr(vid, attr) and os.path.exists(getattr(vid, attr)):
                return getattr(vid, attr)
        if isinstance(vid, dict):
            for k in ("video_path", "filename", "path"):
                if k in vid and os.path.exists(vid[k]):
                    return vid[k]
        if hasattr(vid, "save_to"):
            tmp = Path(tempfile.mkdtemp(prefix="looptrim_src_")) / "src.mp4"
            vid.save_to(str(tmp))
            return str(tmp)
        return None
