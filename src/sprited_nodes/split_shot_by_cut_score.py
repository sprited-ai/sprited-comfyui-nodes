# comfy_custom_nodes/shot_split_by_cut_score.py
from __future__ import annotations
import math
import numpy as np

# ComfyUI types
import torch

try:
    from skimage.metrics import structural_similarity as ssim_metric
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

from typing import List, Tuple

def _to_numpy_frames(image_batch) -> List[np.ndarray]:
    """
    Accepts ComfyUI IMAGE (torch tensor [N,H,W,C] in 0..1) and returns list of uint8 HxWx3.
    """
    if isinstance(image_batch, torch.Tensor):
        t = image_batch
    elif isinstance(image_batch, dict) and isinstance(image_batch.get("images"), torch.Tensor):
        t = image_batch["images"]
    else:
        # ComfyUI core IMAGE is typically torch.Tensor already; try best-effort.
        t = torch.as_tensor(image_batch)

    # Ensure [N,H,W,C] float 0..1
    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.0 + 1e-6:
        # Assume 0..255; normalize
        t = t / 255.0
    # move to cpu
    t = t.detach().cpu()
    if t.ndim == 3:
        # single image -> [1,H,W,C]
        t = t.unsqueeze(0)
    # Convert each to uint8 numpy
    frames = (t.clamp(0,1).numpy() * 255.0).astype(np.uint8)
    return [frames[i] for i in range(frames.shape[0])]

def _to_image_batch(frames: List[np.ndarray]) -> torch.Tensor:
    """
    frames: list of uint8 HxWx3 -> returns ComfyUI IMAGE tensor [N,H,W,C] float32 0..1
    """
    if not frames:
        # Return an empty 0-length batch with 1x1 shape (ComfyUI tolerates empty poorly).
        return torch.zeros((0,1,1,3), dtype=torch.float32)
    arr = np.stack(frames, axis=0)  # [N,H,W,3], uint8
    t = torch.from_numpy(arr).float() / 255.0
    return t

def _rgb_to_gray(x: np.ndarray) -> np.ndarray:
    # x: HxWx3 uint8
    # Using ITU-R BT.601 luma
    return (0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]).astype(np.float32)

def _frame_pair_score(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """
    Larger score => more likely a cut between a and b.
    """
    if metric == "l2":
        # Mean squared difference in gray
        ga, gb = _rgb_to_gray(a), _rgb_to_gray(b)
        d = ga - gb
        return float((d*d).mean())
    elif metric == "hist":
        # 3x256 histograms, chi-square distance
        score = 0.0
        for c in range(3):
            ha, _ = np.histogram(a[...,c].ravel(), bins=256, range=(0,255), density=True)
            hb, _ = np.histogram(b[...,c].ravel(), bins=256, range=(0,255), density=True)
            # chi-square; add epsilon to avoid division by zero
            eps = 1e-12
            num = (ha - hb)**2
            den = ha + hb + eps
            score += float((num / den).sum())
        return score
    elif metric == "ssim":
        # 1 - SSIM(gray). Requires scikit-image
        if not _HAS_SKIMAGE:
            # Fallback to l2
            return _frame_pair_score(a, b, "l2")
        ga, gb = _rgb_to_gray(a), _rgb_to_gray(b)
        # Normalize to 0..1 for SSIM
        ga1 = (ga / 255.0).clip(0,1)
        gb1 = (gb / 255.0).clip(0,1)
        val = ssim_metric(ga1, gb1, data_range=1.0)
        return float(1.0 - val)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _sliding_window_scores(frames: List[np.ndarray], window: int, metric: str) -> np.ndarray:
    """
    Compute per-transition scores s[t] for cut between frame t and t+1.
    We aggregate in a symmetric window around t: average of pairwise scores
    (t-w..t+w) vs (t-w+1..t+w+1), clamped to valid range.
    """
    n = len(frames)
    if n < 2:
        return np.zeros((0,), dtype=np.float32)

    base = np.zeros((n-1,), dtype=np.float32)
    # precompute adjacent diffs
    for t in range(n-1):
        base[t] = _frame_pair_score(frames[t], frames[t+1], metric)

    if window <= 0:
        return base

    # Smoothed score: average over a window of adjacent diffs around t
    smooth = np.zeros_like(base)
    for t in range(n-1):
        a = max(0, t - window)
        b = min(n-1, t + window)  # inclusive index in base
        smooth[t] = float(base[a:b+1].mean())
    return smooth

def _pick_top_k_minus_1(scores: np.ndarray, k: int, nms_radius: int, min_len: int, n_frames: int) -> List[int]:
    """
    Pick top k-1 cut indices in [0..n-2], with:
      - non-max suppression radius (±nms_radius) to avoid nearby duplicates
      - enforcing min shot length (in frames) between consecutive cuts
    Returns sorted cut indices.
    """
    if k <= 1 or scores.size == 0:
        return []

    # candidate indices sorted by score descending
    idxs = np.argsort(-scores)
    chosen = []
    for t in idxs:
        if len(chosen) >= k-1:
            break

        # NMS: skip if near an already chosen cut
        if any(abs(t - c) <= nms_radius for c in chosen):
            continue

        # Tentatively include and check min_len feasibility
        temp = sorted(chosen + [t])
        cuts = [-1] + temp + [n_frames-1]
        ok = True
        for i in range(len(cuts)-1):
            seg_len = cuts[i+1] - cuts[i]
            if seg_len < min_len:
                ok = False
                break
        if ok:
            chosen.append(int(t))

    chosen.sort()
    return chosen

class ShotSplitByCutScore:
    """
    Split an IMAGE sequence into k shots by selecting k-1 highest cut-likelihood transitions.
    If cut likelihoods are weak (below cut_threshold), duplicate the last segment until k.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "k": ("INT", {"default": 3, "min": 1, "max": 100}),
                "window": ("INT", {"default": 2, "min": 0, "max": 20}),
                "metric": (["l2", "ssim", "hist"], {"default": "l2"}),
                "nms_radius": ("INT", {"default": 2, "min": 0, "max": 60}),
                "min_len": ("INT", {"default": 8, "min": 1, "max": 2000}),
                # NEW: threshold below which we consider transitions "not real"
                "cut_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("shot_images", "cut_indices",)
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "split"
    CATEGORY = "Image/Sequence"

    def split(self, images, k, window, metric, nms_radius, min_len, cut_threshold):
        frames = _to_numpy_frames(images)
        n = len(frames)

        if n == 0:
            print("[ShotSplitByCutScore] Empty input batch.")
            # Return k empty batches
            empty = _to_image_batch([])
            return ([empty for _ in range(max(1, k))], [])

        # Each segment must have ≥1 frame → cannot exceed n
        k = max(1, min(k, n))

        # Scores for transitions between t and t+1 (length n-1)
        scores = _sliding_window_scores(frames, window=window, metric=metric)

        # If there are no transitions (n<2) or everything is too flat, duplicate whole seq.
        if scores.size == 0 or (scores.max() < cut_threshold):
            print(f"[ShotSplitByCutScore] max score too low ({scores.max() if scores.size else 'N/A'}) < {cut_threshold}; repeating full sequence x{k}.")
            full = _to_image_batch(frames)
            return ([full] * k, [])

        # Peak-pick (NMS + min_len feasibility)
        raw_cuts = _pick_top_k_minus_1(scores, k=k, nms_radius=nms_radius, min_len=min_len, n_frames=n)

        # Filter by cut_threshold (only keep strong-enough transitions)
        cuts = [c for c in raw_cuts if scores[c] >= cut_threshold]

        # Build segments from the remaining cuts
        boundaries = [-1] + cuts + [n - 1]  # inclusive end
        shot_batches = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1
            end = boundaries[i + 1]
            if end < start:
                start, end = end, end
            seg = frames[start : end + 1]
            shot_batches.append(_to_image_batch(seg))

        # If we still have fewer than k segments, repeat the last segment
        if len(shot_batches) < k:
            last = shot_batches[-1] if shot_batches else _to_image_batch(frames)
            while len(shot_batches) < k:
                shot_batches.append(last)   # use same tensor reference; .clone() if deep copy desired

        # Do not invent indices for duplicated segments; return only true cut indices that passed threshold
        return (shot_batches, cuts)
