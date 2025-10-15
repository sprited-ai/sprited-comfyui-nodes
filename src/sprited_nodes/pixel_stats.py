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
    """Compute global weighted mean/std across all frames and all color channels."""
    all_pixels = []
    all_weights = []

    for rgb, weight in zip(rgb_list, weight_list):
        h, w, c = rgb.shape
        # Flatten spatial dimensions and stack all channels
        pixels = rgb.reshape(-1)  # Shape: (h*w*c,)
        weights = np.broadcast_to(weight[..., None], (h, w, c)).reshape(-1)  # Shape: (h*w*c,)

        all_pixels.append(pixels)
        all_weights.append(weights)

    # Concatenate all frames
    all_pixels = np.concatenate(all_pixels)
    all_weights = np.concatenate(all_weights)

    # Compute global weighted statistics
    total_weight = np.sum(all_weights) + 1e-8
    global_mean = np.sum(all_pixels * all_weights) / total_weight
    global_var = np.sum(all_weights * (all_pixels - global_mean) ** 2) / total_weight
    global_std = np.sqrt(global_var)

    return float(global_mean), float(global_std)

class PixelRGBStats:
    """
    Computes global weighted mean and std across all frames and all color channels.
    The mask values [0–1] act as weights; if omitted, all pixels are weighted equally.
    Returns:
        mean = single number (global mean)
        std  = single number (global std)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("mean", "std",)
    FUNCTION = "run"
    CATEGORY = "SpriteDX/Analysis"

    def run(self, image, mask=None):
        rgb_list = image_tensor_to_rgb_list(image)
        mask_list = mask_tensor_to_weight_list(mask, rgb_list)

        global_mean, global_std = weighted_global_mean_std(rgb_list, mask_list)

        return (global_mean, global_std)

NODE_CLASS_MAPPINGS = {"PixelRGBStats": PixelRGBStats}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelRGBStats": "Pixel RGB Stats"}
