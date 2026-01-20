"""Anti-Corruption Model Node for ComfyUI

Restores corrupted sprite animations using a trained U-Net model.
The model removes upscaling artifacts, blur, noise, and reconstructs clean RGBA sprites.

Model: sprited/sprite-dx-anti-corruption-v1 (revision: v1.1.1)
Input: 128x128 RGBA (corrupted sprite frames)
Output: 128x128 RGBA (clean sprite frames with transparency)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import tempfile
from PIL import Image

# ============================================================================
# PREPROCESSING HELPERS (from training code)
# ============================================================================

def resize_and_pad(image_tensor, target_size=128):
    """
    Resize image tensor to fit within target_size while maintaining aspect ratio,
    then pad to make it square.

    Args:
        image_tensor: Tensor in [B, C, H, W] format (NCHW) or [C, H, W] format
        target_size: Target size for both dimensions (default: 128)

    Returns:
        padded_tensor: Tensor of shape [B, C, target_size, target_size] or [C, target_size, target_size]
        padding_info: Dict with 'padding' (left, top, right, bottom) and 'original_size'
    """
    # Handle both batched [B, C, H, W] and single [C, H, W] inputs
    is_batched = len(image_tensor.shape) == 4
    if not is_batched:
        image_tensor = image_tensor.unsqueeze(0)

    batch_size, channels, height, width = image_tensor.shape

    # Calculate aspect ratio preserving size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    # Ensure dimensions are at least 1
    new_height = max(1, new_height)
    new_width = max(1, new_width)

    # Resize using bilinear interpolation
    resized = F.interpolate(
        image_tensor,
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )

    # Calculate padding to make it square
    pad_height = target_size - new_height
    pad_width = target_size - new_width

    # Distribute padding evenly (top/bottom, left/right)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding (PyTorch padding order: left, right, top, bottom)
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    padding_info = {
        'padding': (pad_left, pad_top, pad_right, pad_bottom),
        'original_size': (height, width),
        'resized_size': (new_height, new_width)
    }

    if not is_batched:
        padded = padded.squeeze(0)

    return padded, padding_info

# ============================================================================
# U-NET ARCHITECTURE (from training code)
# ============================================================================

class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample + Concat + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AntiCorruptionUNet(nn.Module):
    """
    U-Net for anti-corruption: RGBA (4 channels) -> RGBA (4 channels)
    Input: 128x128x4 (corrupted RGBA, alpha may be all 1s or corrupted)
    Output: 128x128x4 (clean RGBA)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(4, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck
        self.down4 = Down(512, 512)

        # Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)

        # Output heads
        self.out_rgb = nn.Conv2d(64, 3, 1)
        self.out_alpha = nn.Conv2d(64, 1, 1)

        # Initialize output layers for identity-like behavior
        # RGB: small weights so residual corrections are small
        nn.init.normal_(self.out_rgb.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.out_rgb.bias)

        # Alpha: keep random initialization
        nn.init.normal_(self.out_alpha.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.out_alpha.bias)

    def forward(self, x):
        input_rgba = x  # Save input for residual connection (RGBA)

        # Encoder with skip connections
        x1 = self.inc(x)      # 128x128x64
        x2 = self.down1(x1)   # 64x64x128
        x3 = self.down2(x2)   # 32x32x256
        x4 = self.down3(x3)   # 16x16x512
        x5 = self.down4(x4)   # 8x8x512

        # Decoder with skip connections
        x = self.up1(x5, x4)  # 16x16x256
        x = self.up2(x, x3)   # 32x32x128
        x = self.up3(x, x2)   # 64x64x64
        x = self.up4(x, x1)   # 128x128x64

        # Separate output heads with residual connection for RGB only
        rgb_residual = self.out_rgb(x)
        rgb = input_rgba[:, :3] + rgb_residual  # Identity + learned corrections for RGB, unbounded
        alpha = torch.sigmoid(self.out_alpha(x))  # Direct prediction for alpha

        return torch.cat([rgb, alpha], dim=1)

    def post_process(self, x):
        """Post-process output to ensure valid RGBA."""
        rgb = torch.clamp(x[:, :3], 0.0, 1.0)
        alpha = torch.clamp(x[:, 3:4], 0.0, 1.0)
        return torch.cat([rgb, alpha], dim=1)


# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

class ModelManager:
    """Manages model downloading, caching, and loading."""

    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cache_dir = Path(tempfile.gettempdir()) / "comfyui_sprite_anticorruption"
            self.cache_dir.mkdir(exist_ok=True, parents=True)

            # Local models directory (checked first, before HuggingFace)
            # Path relative to ComfyUI root: custom_nodes/sprited-comfyui-nodes/src/sprited_nodes -> ../../../..
            comfyui_root = Path(__file__).parent.parent.parent.parent.parent
            self.local_models_dir = comfyui_root / "models" / "sprite_dx_anti_corruption"
            self.local_models_dir.mkdir(exist_ok=True, parents=True)

    def load_model(self, force_reload=False, force_download=False):
        """Load model from local directory or HuggingFace.

        Priority order:
        1. Check models/sprite_dx_anti_corruption/model.safetensors (local)
        2. Download from HuggingFace if not found locally

        Args:
            force_reload: Force reload model into memory (clears cached model)
            force_download: Force re-download from HuggingFace (skips local check)
        """
        if self._model is not None and not force_reload and not force_download:
            return self._model

        print(f"[AntiCorruption] Loading model on {self._device}...")

        model_filename = "model.safetensors"
        repo_id = "sprited/sprite-dx-anti-corruption-v1"
        revision = "v1.1.1"  # Use specific version tag
        model_path = None

        # Check local models directory first (unless force_download is enabled)
        if not force_download:
            local_model_path = self.local_models_dir / model_filename
            if local_model_path.exists():
                print(f"[AntiCorruption] Using local model from: {local_model_path}")
                model_path = str(local_model_path)

        # If not found locally, download from HuggingFace
        if model_path is None:
            downloaded_path = None
            if force_download:
                # Force re-download from HuggingFace, ignore cache
                print(f"[AntiCorruption] Force download enabled, fetching fresh model from HuggingFace...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_filename,
                        revision=revision,
                        cache_dir=str(self.cache_dir),
                        force_download=True  # Force fresh download
                    )
                    print(f"[AntiCorruption] Model downloaded to cache: {downloaded_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download model from HuggingFace: {str(e)}")
            else:
                try:
                    # Check if model is already cached by HuggingFace
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_filename,
                        revision=revision,
                        cache_dir=str(self.cache_dir),
                        local_files_only=True  # Don't download, only check cache
                    )
                    print(f"[AntiCorruption] Using HuggingFace cached model from: {downloaded_path}")

                except Exception:
                    # Model not in cache, download it
                    print(f"[AntiCorruption] Model not found locally or in cache, downloading from HuggingFace...")
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=model_filename,
                            revision=revision,
                            cache_dir=str(self.cache_dir)
                        )
                        print(f"[AntiCorruption] Model downloaded to cache: {downloaded_path}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to download model from HuggingFace: {str(e)}")

            # Copy downloaded model to local models directory for future use
            if downloaded_path:
                import shutil
                local_model_path = self.local_models_dir / model_filename
                try:
                    shutil.copy2(downloaded_path, local_model_path)
                    print(f"[AntiCorruption] Copied model to local directory: {local_model_path}")
                    model_path = str(local_model_path)
                except Exception as e:
                    print(f"[AntiCorruption] Warning: Failed to copy model to local directory: {e}")
                    print(f"[AntiCorruption] Using cached version: {downloaded_path}")
                    model_path = downloaded_path
        try:
            # Create model architecture
            model = AntiCorruptionUNet().to(self._device)

            # Load weights from safetensors
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict)

            model.eval()
            self._model = model

            print(f"[AntiCorruption] Model loaded successfully!")
            return self._model

        except Exception as e:
            raise RuntimeError(f"Failed to load anti-corruption model weights: {str(e)}")

    @property
    def device(self):
        return self._device


# ============================================================================
# COMFYUI NODE
# ============================================================================

class SpriteAntiCorruptionNode:
    """
    ComfyUI node for sprite animation restoration.

    Takes corrupted RGB sprite frames and outputs clean RGBA frames.
    Expects input images to be 128x128 (will resize if needed).
    """

    def __init__(self):
        self.model_manager = ModelManager()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI IMAGE tensor [B, H, W, C]
            },
            "optional": {
                "auto_resize": ("BOOLEAN", {"default": True}),
                "force_reload_model": ("BOOLEAN", {"default": False}),
                "force_download_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore"
    CATEGORY = "image/restoration"
    DESCRIPTION = "Restore corrupted sprite animations to clean RGBA using trained U-Net model."

    def restore(self, images, auto_resize=True, force_reload_model=False, force_download_model=False):
        """
        Restore corrupted sprite frames.

        Args:
            images: ComfyUI IMAGE tensor [B, H, W, C] in range [0, 1], RGB or RGBA
                    If RGB, will be converted to RGBA with opaque alpha (all 1s)
                    If RGBA, the model will process both RGB and alpha channels
            auto_resize: Whether to automatically resize to 128x128
            force_reload_model: Force reload model from memory cache
            force_download_model: Force re-download model from HuggingFace (ignores local cache)

        Returns:
            Tuple of (restored_images,) as RGBA IMAGE tensor [B, 128, 128, 4]
        """
        # Load model
        model = self.model_manager.load_model(force_reload=force_reload_model, force_download=force_download_model)
        device = self.model_manager.device

        # Convert ComfyUI IMAGE to torch tensor
        # ComfyUI format: [B, H, W, C] float32 in [0, 1]
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)

        batch_size, height, width, channels = images.shape

        # Convert RGB to RGBA if needed (model expects RGBA)
        if channels == 3:
            # Add opaque alpha channel (all 1s)
            alpha_channel = torch.ones((batch_size, height, width, 1), dtype=images.dtype, device=images.device)
            images = torch.cat([images, alpha_channel], dim=3)
            print(f"[AntiCorruption] Converted RGB to RGBA (added opaque alpha channel)")
        elif channels == 4:
            # Already RGBA, pass through as-is
            print(f"[AntiCorruption] Input is RGBA, passing through to model")
        else:
            raise ValueError(f"Expected 3 or 4 channel input, got {channels}")

        # Use RGBA images for processing
        # (no need to set images = rgb_images since we already have RGBA)

        # Debug: Show transparency statistics BEFORE resize/padding
        alpha_channel_before = images[:, :, :, 3]  # [B, H, W]
        total_pixels_before = alpha_channel_before.numel()
        transparent_pixels_before = (alpha_channel_before < 0.1).sum().item()
        transparent_pct_before = (transparent_pixels_before / total_pixels_before) * 100
        print(f"[AntiCorruption] BEFORE resize - Transparent pixels: {transparent_pct_before:.2f}% ({transparent_pixels_before}/{total_pixels_before})")

        # Check if resize is needed
        needs_resize = (height != 128 or width != 128)

        if needs_resize and not auto_resize:
            raise ValueError(
                f"Input size is {height}x{width}, but model expects 128x128. "
                f"Set auto_resize=True or resize input images."
            )

        if needs_resize:
            print(f"[AntiCorruption] Resizing from {height}x{width} to 128x128 (maintaining aspect ratio)")
            # Convert [B, H, W, C] -> [B, C, H, W] for resize_and_pad
            images_nchw = images.permute(0, 3, 1, 2)
            # Use padding-based resizing (same as training script)
            images_padded, padding_info = resize_and_pad(images_nchw, target_size=128)
            # Convert back to [B, H, W, C]
            images = images_padded.permute(0, 2, 3, 1)
            print(f"[AntiCorruption] Applied padding: {padding_info['padding']} (left, top, right, bottom)")

        # Debug: Show transparency statistics after resize/padding
        alpha_channel = images[:, :, :, 3]  # [B, H, W]
        total_pixels = alpha_channel.numel()
        transparent_pixels = (alpha_channel < 0.1).sum().item()
        transparent_pct = (transparent_pixels / total_pixels) * 100
        print(f"[AntiCorruption] Transparent pixels: {transparent_pct:.2f}% ({transparent_pixels}/{total_pixels})")

        # Process in batches to avoid memory issues
        restored_frames = []

        with torch.no_grad():
            for i in range(batch_size):
                # Get single frame [H, W, C] -> [C, H, W]
                frame = images[i].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                frame = frame.to(device)

                # Run through model
                output = model(frame)

                # Post-process to ensure valid range
                output = model.post_process(output)

                # Convert back to ComfyUI format [H, W, C]
                output_frame = output[0].permute(1, 2, 0).cpu()  # [H, W, 4]
                restored_frames.append(output_frame)

        # Stack back into batch [B, H, W, C]
        restored_batch = torch.stack(restored_frames, dim=0)

        print(f"[AntiCorruption] Restored {batch_size} frames ({restored_batch.shape})")

        return (restored_batch,)


# For backward compatibility / alternative names
class SpriteDXAntiCorruptionV1(SpriteAntiCorruptionNode):
    """SpriteDX Anti-Corruption V1 - Main class"""
    pass


# Keep unversioned alias for backward compatibility
class SpriteDXAntiCorruption(SpriteDXAntiCorruptionV1):
    """Alias for SpriteDXAntiCorruptionV1"""
    pass
