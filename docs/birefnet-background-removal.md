# BiRefNet Background Removal Node

A ComfyUI custom node for removing backgrounds from images using the fine-tuned BiRefNet model (ToonOut).

## Features

- **High-Quality Background Removal**: Uses the ToonOut fine-tuned BiRefNet model for superior results on illustrations and animations
- **Seam Tightening**: Optional post-processing to remove semi-transparent edge artifacts
- **Batch Processing**: Efficiently process multiple images in batches
- **GPU Acceleration**: Automatic CUDA support with fallback to CPU
- **Model Caching**: Model is loaded once and cached for subsequent runs
- **Dual Output**: Returns both RGBA images and separate alpha masks

## Installation

The node is included in the sprited-comfyui-nodes package. Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` and `torchvision` - PyTorch for model inference
- `transformers` - HuggingFace transformers for model loading
- `huggingface_hub` - Auto-download fine-tuned weights
- `opencv-python` - Image processing
- `scipy` - Spatial operations for seam tightening

## Usage

### In ComfyUI

1. Add the "BiRefNet Background Removal (ToonOut) 🌱" node to your workflow
2. Connect an IMAGE input (batch supported)
3. Adjust parameters:
   - **seam_width** (0-10): Width of seam removal, 0 disables tightening
   - **threshold** (0-255): Alpha threshold for tightening, default 200
   - **batch_size** (1-32): Number of images to process simultaneously
   - **weights_path** (optional): Path to custom weights, leave empty to auto-download

### Outputs

- **rgba_images**: IMAGE tensor with 4 channels (RGBA)
- **alpha_mask**: MASK tensor (grayscale alpha channel only)

### Example Workflow

```
LoadImage -> BiRefNetBackgroundRemoval -> SaveImage
                                       -> (use alpha_mask for masking)
```

## Parameters

### seam_width (default: 1)
Controls the width of seam removal for post-processing. Higher values remove more edge artifacts but may affect fine details.
- `0`: No tightening (faster)
- `1-3`: Recommended range for most images
- `4-10`: Aggressive tightening for very noisy masks

### threshold (default: 200)
Alpha threshold for determining which pixels to process during tightening. Pixels with alpha below this value are considered semi-transparent.
- `150-180`: More aggressive, keeps more pixels
- `200`: Balanced (default)
- `220-240`: Conservative, only keeps highly opaque pixels

### batch_size (default: 4)
Number of images to process in one GPU batch. Higher values are faster but use more VRAM.
- `1-2`: Low VRAM mode (< 6GB)
- `4-8`: Standard (6-12GB)
- `16-32`: High throughput (> 12GB)

## Model Information

- **Base Model**: ZhengPeng7/BiRefNet (HuggingFace)
- **Fine-tuned Model**: joelseytre/toonout
- **Auto-download**: Weights are automatically downloaded from HuggingFace on first use
- **Cache Location**: Models are cached by HuggingFace hub in `~/.cache/huggingface/`

## Performance

The node includes performance optimizations:
- Model is loaded once and cached across multiple runs
- Batch processing for multiple images
- GPU acceleration when available
- Efficient tensor operations

Typical performance (RTX 3090):
- Single 512x512 image: ~0.5 seconds
- Batch of 10 images: ~2 seconds
- Memory usage: ~4GB VRAM

## Troubleshooting

### Out of Memory Error
Reduce `batch_size` to 1 or 2

### Model Download Issues
Ensure you have internet connection and HuggingFace hub access. You can manually download weights and specify `weights_path`.

### Slow Performance
- Verify CUDA is available: the node will print "Using device: cuda" or "Using device: cpu"
- Increase `batch_size` if you have available VRAM
- Disable seam tightening by setting `seam_width=0`

## Testing

Run the test script to verify installation:

```bash
python test_birefnet.py
```

This will create a test image, process it, and save the results.

## Credits

- BiRefNet model: [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
- ToonOut fine-tuned weights: [joelseytre/toonout](https://huggingface.co/joelseytre/toonout)
- Original script: birefnet.py

## License

See LICENSE file in the repository root.
