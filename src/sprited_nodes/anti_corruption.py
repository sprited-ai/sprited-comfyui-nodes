"""
Anti-Corruption Model Node for ComfyUI

Restores corrupted sprite animations using a trained U-Net model.
The model removes upscaling artifacts, blur, noise, and reconstructs clean RGBA sprites.

Model: sprited/sprite-dx-anti-corruption-v1
Input: 128x128 RGB (corrupted sprite frames)
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
    U-Net for anti-corruption: RGB (3 channels) -> RGBA (4 channels)
    Input: 128x128x3 (corrupted RGB)
    Output: 128x128x4 (clean RGBA)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(3, 64)
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

    def forward(self, x):
        input_rgb = x  # Save input for residual connection

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

        # Separate output heads with residual connection for RGB
        rgb_residual = self.out_rgb(x)
        rgb = input_rgb + rgb_residual  # Residual connection
        alpha = torch.sigmoid(self.out_alpha(x))

        return torch.cat([rgb, alpha], dim=1)

    def post_process(self, x):
        """Post-process output to ensure valid RGBA range [0, 1]."""
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

    def load_model(self, force_reload=False):
        """Load model from HuggingFace, using cache if available."""
        if self._model is not None and not force_reload:
            return self._model

        print(f"[AntiCorruption] Loading model on {self._device}...")

        try:
            # Check if model is already cached locally
            model_filename = "model.safetensors"
            repo_id = "sprited/sprite-dx-anti-corruption-v1"

            # Try to find cached model first
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                cache_dir=str(self.cache_dir),
                local_files_only=True  # Don't download, only check cache
            )
            print(f"[AntiCorruption] Using cached model from: {model_path}")

        except Exception:
            # Model not in cache, download it
            print(f"[AntiCorruption] Model not found in cache, downloading from HuggingFace...")
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename,
                    cache_dir=str(self.cache_dir)
                )
                print(f"[AntiCorruption] Model downloaded to: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model from HuggingFace: {str(e)}")

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_images",)
    FUNCTION = "restore"
    CATEGORY = "image/restoration"
    DESCRIPTION = "Restore corrupted sprite animations to clean RGBA using trained U-Net model."

    def restore(self, images, auto_resize=True, force_reload_model=False):
        """
        Restore corrupted sprite frames.

        Args:
            images: ComfyUI IMAGE tensor [B, H, W, C] in range [0, 1], RGB
            auto_resize: Whether to automatically resize to 128x128
            force_reload_model: Force reload model from disk

        Returns:
            Tuple of (restored_images,) as RGBA IMAGE tensor [B, 128, 128, 4]
        """
        # Load model
        model = self.model_manager.load_model(force_reload=force_reload_model)
        device = self.model_manager.device

        # Convert ComfyUI IMAGE to torch tensor
        # ComfyUI format: [B, H, W, C] float32 in [0, 1]
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)

        batch_size, height, width, channels = images.shape

        # Ensure RGB (drop alpha if present)
        if channels == 4:
            images = images[:, :, :, :3]
            print(f"[AntiCorruption] Dropped alpha channel from input (had RGBA, using RGB)")
        elif channels != 3:
            raise ValueError(f"Expected 3 or 4 channel input, got {channels}")

        # Check if resize is needed
        needs_resize = (height != 128 or width != 128)

        if needs_resize and not auto_resize:
            raise ValueError(
                f"Input size is {height}x{width}, but model expects 128x128. "
                f"Set auto_resize=True or resize input images."
            )

        if needs_resize:
            print(f"[AntiCorruption] Resizing from {height}x{width} to 128x128")
            # Resize using bilinear interpolation
            # Convert [B, H, W, C] -> [B, C, H, W] for F.interpolate
            images_nchw = images.permute(0, 3, 1, 2)
            images_resized = F.interpolate(
                images_nchw,
                size=(128, 128),
                mode='bilinear',
                align_corners=False
            )
            # Convert back to [B, H, W, C]
            images = images_resized.permute(0, 2, 3, 1)

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
