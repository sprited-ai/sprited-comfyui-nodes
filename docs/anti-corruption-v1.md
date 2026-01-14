# Sprite Anti-Corruption V1 - Implementation Notes

**Date:** January 13, 2026
**Model:** `sprited/sprite-dx-anti-corruption-v1` (HuggingFace)
**Purpose:** Restore corrupted sprite animations to clean RGBA using trained U-Net

---

## Overview

This node implements a sprite animation restoration system that removes artifacts from upscaling, blur, noise, and JPEG compression. It also reconstructs proper alpha channel transparency.

### Training Details

The model was trained with:
- **Architecture:** U-Net with residual RGB connection
- **Input:** 128x128 RGB (corrupted sprite frames)
- **Output:** 128x128 RGBA (clean sprite frames)
- **Training samples:** 1000 per epoch with online augmentation
- **Validation:** 200 cached samples (deterministic)
- **Corruption pipeline:**
  - Random upscaling (2-3x nearest neighbor)
  - Gaussian noise (0.004-0.016 per upscale level)
  - Gaussian blur (0.0-2.5 radius)
  - Bilinear downscaling
  - Translation shifts
  - Composited on white/magenta background

### Loss Function

The training used a weighted loss combining:
1. **RGB Loss:** Weighted by alpha mask dilation and local contrast
2. **Alpha Loss:** Weighted by whiteness (to handle white edges near transparency)
3. **Alpha weight:** 2.9x (emphasizes alpha reconstruction)

---

## Implementation

### Architecture Components

The U-Net consists of:

```
Encoder:
- inc: DoubleConv(3→64)
- down1: Down(64→128)    @ 64x64
- down2: Down(128→256)   @ 32x32
- down3: Down(256→512)   @ 16x16
- down4: Down(512→512)   @ 8x8 (bottleneck)

Decoder:
- up1: Up(512→256)       @ 16x16
- up2: Up(256→128)       @ 32x32
- up3: Up(128→64)        @ 64x64
- up4: Up(64→64)         @ 128x128

Output:
- out_rgb: Conv2d(64→3)
- out_alpha: Conv2d(64→1) + sigmoid
- RGB uses residual connection (input + learned corrections)
```

### Key Design Decisions

1. **Residual RGB:** `rgb = input_rgb + out_rgb(x)`
   - Allows model to learn corrections rather than full reconstruction
   - Initialized with very small weights (std=0.001) for near-identity start
   - Unbounded during training for full gradient flow
   - Clamped to [0,1] during post-processing

2. **Sigmoid Alpha:** Direct sigmoid output for alpha channel
   - Binary transparency during training (0 or 255)
   - Smooth gradients during inference

3. **Skip Connections:** Standard U-Net skip connections preserve spatial details

### Model Loading

Uses singleton pattern (`ModelManager`) to:
- Cache model once in memory
- Download from HuggingFace Hub on first use
- Store in system temp directory: `{tempdir}/comfyui_sprite_anticorruption/`
- Support force reload for debugging

### ComfyUI Integration

**Input:**
- `images`: IMAGE tensor `[B, H, W, C]` in range `[0, 1]`, RGB or RGBA
- `auto_resize`: Boolean (default: True) - resize to 128x128 if needed
- `force_reload_model`: Boolean (default: False) - force model reload

**Output:**
- `restored_images`: IMAGE tensor `[B, 128, 128, 4]` RGBA in range `[0, 1]`

**Processing:**
1. Drop alpha channel if input is RGBA (model expects RGB)
2. Resize to 128x128 using bilinear interpolation if needed
3. Process frames individually through model (avoid VRAM spikes)
4. Post-process to clamp values to valid range
5. Return as ComfyUI IMAGE tensor

---

## Usage Patterns

### Basic Workflow
```
Load Images (128x128 RGB)
    ↓
Sprite Anti-Corruption 🌱
    ↓
Preview Image (now RGBA)
```

### Video Frame Restoration
```
Load Video
    ↓
Video Shot Splitter 🌱
    ↓
Sprite Anti-Corruption 🌱 (batch process)
    ↓
Save Images
```

### Upscaled Input
```
Load Images (256x256 or larger)
    ↓
Sprite Anti-Corruption 🌱 (auto_resize=True)
    ↓
[Restored to 128x128 RGBA]
    ↓
Upscale Image (if desired)
```

---

## Technical Details

### Memory Considerations
- Model size: ~50M parameters (~200MB on disk as safetensors)
- Processing: One frame at a time to avoid memory issues
- Batch dimension preserved but processed sequentially

### Device Selection
- Automatically uses CUDA if available
- Falls back to CPU (slower but works)
- Device is singleton (one model instance per ComfyUI session)

### Transparency Handling

The model was trained with specific transparency preprocessing:
1. **Eroded RGB values:** Transparent pixels get RGB from neighboring opaque pixels
   - Prevents "garbage" RGB values in transparent areas
   - Ensures smooth color boundaries

2. **Binary alpha training:** Only fully transparent (0) or opaque (255)
   - Validation checked for no partial transparency (1-254)
   - Simplified the learning problem

3. **Whiteness weighting:** Alpha loss amplified near white pixels
   - Addresses "white edge corruption" from compositing
   - Critical for sprite sheets with white outlines

### Resize Strategy

When input is not 128x128:
- **Downscaling:** Bilinear interpolation (preserves details)
- **Upscaling:** Bilinear interpolation (smooths artifacts before model)
- Model designed for 128x128, so resize is recommended
- Can disable with `auto_resize=False` (will raise error)

---

## Node Registration

Two node names for flexibility:
1. `SpriteAntiCorruptionNode` → "Sprite Anti-Corruption 🌱"
2. `SpriteDXAntiCorruption` → "SpriteDX Anti-Corruption V1 🌱"

Both are aliases of the same implementation.

---

## Dependencies

**New requirements:**
- `huggingface_hub>=0.20.0` - Download models from HuggingFace
- `safetensors>=0.4.0` - Load model weights

**Existing:**
- `torch` (already in requirements)
- `numpy` (already in requirements)

Install with:
```bash
pip install -r requirements.txt
```

---

## Testing Checklist

- [ ] Model downloads from HuggingFace on first run
- [ ] Model caches correctly (no re-download on subsequent runs)
- [ ] Processes single image (1 frame batch)
- [ ] Processes image batch (multiple frames)
- [ ] Auto-resize works for non-128x128 inputs
- [ ] RGBA input drops alpha correctly
- [ ] Output is valid RGBA in [0, 1] range
- [ ] Works on CUDA (if available)
- [ ] Works on CPU fallback
- [ ] Node appears in ComfyUI node browser
- [ ] Integration with other nodes (video splitter, etc.)

---

## Known Limitations

1. **Fixed resolution:** Model only works at 128x128
   - Training data was all 128x128 crops
   - Would need retraining for other resolutions

2. **Sprite-specific:** Optimized for sprite art, not photographs
   - Trained on pixel art / sprite datasets
   - May not generalize to realistic images

3. **Corruption assumptions:** Best for:
   - Upscaling artifacts (nearest neighbor + bilinear)
   - Mild blur and noise
   - Composite transparency issues
   - NOT for: heavy compression, extreme distortion, completely missing data

4. **Alpha reconstruction:** Requires some visual cues
   - Cannot magically know what should be transparent
   - Works best when transparency follows sprite boundaries

---

## Future Improvements

### V2 Model Ideas
- Multi-resolution support (64x64, 128x128, 256x256)
- Attention mechanisms for better long-range dependencies
- Perceptual loss component (LPIPS)
- Adversarial training for sharper outputs
- Fine-tuning on specific sprite styles (RPG, platformer, etc.)

### Node Features
- Batch size parameter (process N frames at a time)
- Strength/blend parameter (mix with original)
- Preview comparison (before/after side-by-side)
- Advanced mode: expose internal hyperparameters
- Chain multiple passes for heavy corruption

### Engineering
- Model quantization (int8) for faster inference
- ONNX export for cross-platform compatibility
- TorchScript compilation for production speedup
- Progress bar for large batches

---

## Training Reproduction

If you need to retrain or fine-tune:

1. Use the training script from the original implementation
2. Key hyperparameters:
   - Learning rate: 0.001 with ReduceLROnPlateau
   - Alpha weight: 2.9
   - Batch size: 8
   - Epochs: 200
   - Optimizer: Adam

3. Data requirements:
   - Clean RGBA sprite images (various sizes)
   - Model will crop 128x128 patches
   - Minimum 10-20 diverse sprites recommended

4. Hardware:
   - GPU with 8GB+ VRAM recommended
   - CPU training is very slow (~100x slower)

---

## Credits

**Model Training:** Jin (Sprite-DX Team)
**Architecture:** U-Net with residual connections
**Framework:** PyTorch + ComfyUI
**Model Host:** HuggingFace Hub

---

## Changelog

### V1 (Current)
- Initial release
- 128x128 RGB → RGBA restoration
- HuggingFace Hub integration
- ComfyUI node implementation
- Automatic caching and device selection
- Auto-resize support

---

## Support

For issues or questions:
- GitHub Issues: `sprited-comfyui-nodes` repository
- HuggingFace Model: https://huggingface.co/sprited/sprite-dx-anti-corruption-v1
- ComfyUI Discord: #custom-nodes channel
