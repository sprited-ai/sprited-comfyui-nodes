# sprited_pixel_rgb_stats_weighted_bare.py
import numpy as np

def image_tensor_to_rgb_list(img_tensor):
    """Convert ComfyUI IMAGE (B,H,W,C) [0,1] -> list of np.float32 RGB arrays"""
    if img_tensor is None:
        return []
    arr = np.asarray(img_tensor, np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return [np.clip(arr[i], 0, 1) for i in range(arr.shape[0])]

def mask_tensor_to_weight_list(mask_tensor, like_images):
    """Convert ComfyUI MASK (B,H,W) [0,1] -> list of float32 arrays [H,W]"""
    if mask_tensor is None:
        return [np.ones(img.shape[:2], np.float32) for img in like_images]
    m = np.asarray(mask_tensor, np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    weights = []
    for i, img in enumerate(like_images):
        mi = m[i] if i < m.shape[0] else m[-1]
        if mi.ndim == 3 and mi.shape[-1] == 3:
            # collapse RGB mask to single channel
            mi = mi.mean(axis=2)
        weights.append(np.clip(mi, 0.0, 1.0))
    return weights

def weighted_global_mean_std(rgb_list, weight_list):
    """Compute global weighted mean/std across all frames using grayscale luminance."""
    all_pixels = []
    all_weights = []

    for rgb, weight in zip(rgb_list, weight_list):
        h, w, c = rgb.shape
        # Convert RGB to grayscale using ITU-R BT.601 formula
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]  # Shape: (h, w)

        # Flatten spatial dimensions
        pixels = gray.reshape(-1)  # Shape: (h*w,)
        weights = weight.reshape(-1)  # Shape: (h*w,)

        # Filter out masked pixels (weight < threshold)
        mask = weights > 1e-6
        pixels = pixels[mask]
        weights = weights[mask]

        if len(pixels) > 0:
            all_pixels.append(pixels)
            all_weights.append(weights)

    if len(all_pixels) == 0:
        return 0.0, 0.0

    # Concatenate all frames
    all_pixels = np.concatenate(all_pixels)
    all_weights = np.concatenate(all_weights)

    # Compute global weighted statistics
    total_weight = np.sum(all_weights) + 1e-8
    global_mean = np.sum(all_pixels * all_weights) / total_weight
    global_var = np.sum(all_weights * (all_pixels - global_mean) ** 2) / total_weight
    global_std = np.sqrt(global_var)

    return float(global_mean), float(global_std)

def weighted_per_frame_mean_std(rgb_list, weight_list):
    """Compute per-frame weighted mean/std using grayscale luminance for each frame."""
    frame_means = []
    frame_stds = []

    for rgb, weight in zip(rgb_list, weight_list):
        h, w, c = rgb.shape
        # Convert RGB to grayscale using ITU-R BT.601 formula
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]  # Shape: (h, w)

        # Flatten spatial dimensions
        pixels = gray.reshape(-1)  # Shape: (h*w,)
        weights = weight.reshape(-1)  # Shape: (h*w,)

        # Filter out masked pixels (weight < threshold)
        mask = weights > 1e-6
        pixels = pixels[mask]
        weights = weights[mask]

        if len(pixels) == 0:
            frame_means.append(0.0)
            frame_stds.append(0.0)
            continue

        # Compute weighted statistics for this frame
        total_weight = np.sum(weights) + 1e-8
        frame_mean = np.sum(pixels * weights) / total_weight
        frame_var = np.sum(weights * (pixels - frame_mean) ** 2) / total_weight
        frame_std = np.sqrt(frame_var)

        frame_means.append(float(frame_mean))
        frame_stds.append(float(frame_std))

    return frame_means, frame_stds

class PixelRGBStats:
    """
    Computes both global and per-frame weighted mean and std using grayscale luminance.
    RGB is converted to grayscale (ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B) before computing statistics.
    The mask values [0–1] act as weights; if omitted, all pixels are weighted equally.
    Returns:
        global_mean = single number (global grayscale mean across all frames)
        global_std  = single number (global grayscale std across all frames)
        frame_means = list of numbers (grayscale mean for each frame)
        frame_stds  = list of numbers (grayscale std for each frame)
        max_mean    = single number (maximum frame mean)
        max_std     = single number (maximum frame std)
        min_mean    = single number (minimum frame mean)
        min_std     = single number (minimum frame std)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "LIST", "LIST", "FLOAT", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("global_mean", "global_std", "frame_means", "frame_stds", "max_mean", "max_std", "min_mean", "min_std",)
    FUNCTION = "run"
    CATEGORY = "SpriteDX/Analysis"

    def run(self, image, mask=None):
        rgb_list = image_tensor_to_rgb_list(image)
        mask_list = mask_tensor_to_weight_list(mask, rgb_list)

        # Compute global statistics
        global_mean, global_std = weighted_global_mean_std(rgb_list, mask_list)

        # Compute per-frame statistics
        frame_means, frame_stds = weighted_per_frame_mean_std(rgb_list, mask_list)

        # Compute max and min statistics
        max_mean = float(max(frame_means)) if frame_means else 0.0
        max_std = float(max(frame_stds)) if frame_stds else 0.0
        min_mean = float(min(frame_means)) if frame_means else 0.0
        min_std = float(min(frame_stds)) if frame_stds else 0.0

        return (global_mean, global_std, frame_means, frame_stds, max_mean, max_std, min_mean, min_std)

NODE_CLASS_MAPPINGS = {"PixelRGBStats": PixelRGBStats}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelRGBStats": "Pixel RGB Stats"}
