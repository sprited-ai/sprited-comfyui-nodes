"""
LoopMomentumNode – Custom ComfyUI node for seamless loop extraction maximizing conservation of momentum
"""
import json, tempfile, math, os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import imagehash

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    VideoFromFile = str

def phash_vec(img):
    return imagehash.phash(img).hash.astype(np.uint8).flatten()

def frame_vec(img, size=16):
    return np.array(
        img.convert("L").resize((size, size), Image.NEAREST),
        dtype=np.float32
    ).flatten()

def compute_optical_flow(frames):
    flows = []
    prev = np.array(frames[0].convert("L"), dtype=np.uint8)
    for i in range(1, len(frames)):
        curr = np.array(frames[i].convert("L"), dtype=np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev = curr
    return flows

def avg_motion_vector(flow):
    return np.mean(flow, axis=(0, 1))

def score_momentum_conservation(flows, i, j):
    vi = avg_motion_vector(flows[i])
    vj = avg_motion_vector(flows[j])
    return np.linalg.norm(vi - vj)

def frame_distance(a, b):
    return np.count_nonzero(a ^ b)

def score_segment(hashes, flows, start, length, lam1=0.5, lam2=1.0):
    end = start + length
    seam = frame_distance(hashes[start], hashes[end])
    momentum_diff = score_momentum_conservation(flows, start, end-1)
    total = seam + lam1 * (1.0 / (np.abs(momentum_diff) + 1e-6)) + lam2 * momentum_diff
    return total, seam, momentum_diff

def detect_best_loop(frames, min_len=20, max_len=120, lam1=0.5, lam2=1.0):
    hashes = [phash_vec(f) for f in frames]
    flows = compute_optical_flow(frames)
    best = (np.inf, 0, 0, 0, 0)
    for length in range(min_len, min(max_len, len(frames)-2)):
        for start in range(0, len(frames) - length - 1):
            total, seam, momentum_diff = score_segment(hashes, flows, start, length, lam1, lam2)
            if total < best[0]:
                best = (total, start, start + length, seam, momentum_diff)
    return best  # total, start, end, seam, momentum_diff

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
    else:
        import imageio.v3 as iio
        arr = [np.array(f) for f in sliced]
        iio.imwrite(out_path, arr, fps=fps,
                    codec="libx264", quality=8, pixelformat="yuv420p")

class LoopMomentumNode:
    """
    Detects the best seamless loop inside an animated clip by maximizing conservation of momentum.
    Outputs the trimmed VIDEO and a JSON summary.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "min_gap":    ("INT", {"default": 10, "min": 2,  "max": 240}),
                "max_gap":    ("INT", {"default": 120,"min": 10, "max": 600}),
                "lambda_1":   ("FLOAT", {"default": 0.5,"min": 0.0, "max": 5.0, "step": 0.1}),
                "lambda_2":   ("FLOAT", {"default": 1.0,"min": 0.0, "max": 5.0, "step": 0.1}),
                "format":     (["mp4", "gif", "webp"], {"default": "webp"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES     = ("VIDEO", "STRING")
    RETURN_NAMES     = ("trimmed_video", "info_json")
    CATEGORY         = "Video/Edit"
    FUNCTION         = "run"

    def run(self, video, min_gap, max_gap, lambda_1, lambda_2, format, output_dir=""):
        src = self._resolve_path(video)
        if not src:
            raise RuntimeError("Cannot resolve input video path.")
        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="loopmomentum_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(src).stem}_momentum_loop.{format}"
        # Load frames
        fps = 12.0
        try:
            if src.lower().endswith('.webp'):
                img = Image.open(src)
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
                import imageio.v3 as iio
                try:
                    meta = iio.immeta(src, plugin="FFMPEG")
                    fps = meta.get("fps", fps)
                    frames_np = iio.imread(src, plugin="FFMPEG")
                except Exception:
                    frames_np = iio.imread(src)
                if frames_np.ndim == 3:
                    frames_np = frames_np[None, ...]
                frames = [Image.fromarray(f) for f in frames_np]
        except Exception as e:
            raise RuntimeError(f"Could not read {src}: {e}")
        # Find best loop
        total, start, end, seam, momentum_diff = detect_best_loop(
            frames, min_len=min_gap, max_len=max_gap, lam1=lambda_1, lam2=lambda_2
        )
        info = {
            "start": start,
            "end": end,
            "length": end - start,
            "seam": seam,
            "momentum_diff": round(float(momentum_diff), 6),
            "score": round(float(total), 6),
            "fps": fps
        }
        save_trimmed(frames, start, end - 1, str(out_path), fps)
        info["saved"] = True
        def to_builtin(o):
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        return (VideoFromFile(str(out_path)), json.dumps(info, indent=2, default=to_builtin))

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
            tmp = Path(tempfile.mkdtemp(prefix="loopmomentum_src_")) / "src.mp4"
            vid.save_to(str(tmp))
            return str(tmp)
        return None
