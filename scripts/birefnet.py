#!/usr/bin/env python3
"""
ToonOut Demo Script
Remove background from images using fine-tuned BiRefNet model.

Usage:
    python inference.py --weights path/to/weights.pth --input path/to/image.jpg [--output result.png]
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
import sys
import os
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
from scipy.spatial import cKDTree
import time

# Simple fix for BiRefNet compatibility
import transformers.configuration_utils
original_getattribute = transformers.configuration_utils.PretrainedConfig.__getattribute__

def patched_getattribute(self, key):
    if key == 'is_encoder_decoder':
        return False
    return original_getattribute(self, key)

transformers.configuration_utils.PretrainedConfig.__getattribute__ = patched_getattribute

from transformers import AutoModelForImageSegmentation


def apply_tightening(image, seam_width=1, threshold=200):
    """
    Apply seam removal to background-matted image.
    """
    # Split into RGB and alpha
    rgb = image.convert('RGB')
    alpha = image.getchannel('A')

    # Apply threshold
    alpha_np = np.array(alpha)
    alpha_np = np.where(alpha_np >= threshold, 255, 0).astype(np.uint8)
    alpha = Image.fromarray(alpha_np)

    # Convert alpha to numpy array
    alpha_float = alpha_np.astype(np.float32) / 255.0

    # Find semi-transparent edge pixels
    edge_mask = ((alpha_float > 0.05) & (alpha_float < 0.95)).astype(np.uint8) * 255

    # Create binary mask for fully opaque pixels only
    _, binary_mask = cv2.threshold(alpha_np, 254, 255, cv2.THRESH_BINARY)

    # Get inner safe region
    kernel = np.ones((3, 3), np.uint8)
    inner_region = cv2.erode(binary_mask, kernel, iterations=seam_width)

    # Seam mask
    seam_mask = cv2.bitwise_or(edge_mask, binary_mask - inner_region)

    # Get coordinates
    seam_coords = np.argwhere(seam_mask > 0)
    inner_coords = np.argwhere(inner_region > 0)

    if len(inner_coords) == 0 or len(seam_coords) == 0:
        print("No seams to fix, returning original")
        result = rgb.copy()
        result.putalpha(alpha)
        return result

    # Process seams
    img_np = np.array(rgb).astype(np.float32)
    result = img_np.copy()
    alpha_result = alpha_float.copy()

    tree = cKDTree(inner_coords)
    search_radius = 1.5

    for seam_px in seam_coords:
        nearby_indices = tree.query_ball_point(seam_px, r=search_radius)

        if len(nearby_indices) == 0:
            dist, idx = tree.query(seam_px, k=1)
            nearby_indices = [idx]

        nearby_coords = inner_coords[nearby_indices]
        dists = np.linalg.norm(nearby_coords - seam_px, axis=1)
        dists = np.where(dists < 1e-6, 1e-6, dists)

        weights = 1.0 / dists
        weights = weights / weights.sum()

        weighted_color = np.zeros(3, dtype=np.float32)
        for w, idx in zip(weights, nearby_indices):
            inner_px = inner_coords[idx]
            weighted_color += w * img_np[inner_px[0], inner_px[1]]

        result[seam_px[0], seam_px[1]] = weighted_color
        alpha_result[seam_px[0], seam_px[1]] = max(alpha_result[seam_px[0], seam_px[1]], 0.98)

    final_result = Image.fromarray(result.astype(np.uint8))
    final_result.putalpha(Image.fromarray((alpha_result * 255).astype(np.uint8)))

    return final_result


def load_birefnet_with_custom_weights(checkpoint_path: str = None):
    """Load BiRefNet model with custom fine-tuned weights"""

    print(f"Loading base BiRefNet model from HuggingFace...")
    # Load the base model from HuggingFace
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    )

    # Download weights from HuggingFace if no local path provided
    if checkpoint_path is None:
        print("Downloading fine-tuned weights from HuggingFace (joelseytre/toonout)...")
        checkpoint_path = hf_hub_download(
            repo_id="joelseytre/toonout",
            filename="birefnet_finetuned_toonout.pth"
        )

    print(f"Loading custom weights from {checkpoint_path}...")
    # Load and apply custom weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Clean up weight keys if needed (remove module prefixes)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module._orig_mod."):
            clean_state_dict[k[len("module._orig_mod."):]] = v
        elif k.startswith("module."):
            clean_state_dict[k[len("module."):]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    print("Model loaded successfully!")
    return model


def remove_background(image_path: str, model, device='cpu', batch_size=4, seam_width=1, threshold=200):
    """Remove background from image using BiRefNet"""

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Processing image: {image_path}")
    # Load image
    img = Image.open(image_path)

    # Check if animated
    is_animated = hasattr(img, 'n_frames') and img.n_frames > 1

    if is_animated:
        print(f"Detected animated image with {img.n_frames} frames")
        print(f"Using batch size: {batch_size}")

        frames_rgba = []
        durations = []

        # Load all frames first
        all_frames = []
        for frame_idx in range(img.n_frames):
            img.seek(frame_idx)
            frame = img.convert('RGB')
            durations.append(img.info.get('duration', 100))
            all_frames.append(frame)

        # Phase 1: Inference - Process all frames in batches
        print(f"\nPhase 1: Running inference on {len(all_frames)} frames...")
        inference_start = time.time()

        # Reset GPU memory stats if using CUDA
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        for batch_start in range(0, len(all_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(all_frames))
            batch_frames = all_frames[batch_start:batch_end]

            print(f"  Inferencing frames {batch_start + 1}-{batch_end}/{len(all_frames)}...")

            # Prepare batch
            batch_tensors = torch.stack([transform(frame) for frame in batch_frames]).to(device)

            # Process batch
            with torch.no_grad():
                preds = model(batch_tensors)[-1].sigmoid().cpu()

            # Convert masks and apply to frames
            for i, (frame, pred) in enumerate(zip(batch_frames, preds)):
                # Convert mask to PIL and resize to original size
                mask = transforms.ToPILImage()(pred.squeeze())
                mask = mask.resize(frame.size)

                # Apply mask to create transparent background
                result_rgba = frame.copy()
                result_rgba.putalpha(mask)
                frames_rgba.append(result_rgba)

        inference_time = time.time() - inference_start

        # Get peak GPU memory usage
        if device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            print(f"✓ Inference completed in {inference_time:.2f}s ({inference_time/len(all_frames):.3f}s per frame)")
            print(f"✓ Peak GPU memory: {peak_memory:.2f}GB")
        else:
            print(f"✓ Inference completed in {inference_time:.2f}s ({inference_time/len(all_frames):.3f}s per frame)")

        # Phase 2: Tightening - Apply to all frames
        print(f"\nPhase 2: Applying tightening to {len(frames_rgba)} frames...")
        tightening_start = time.time()
        frames_tight = []
        for idx, rgba_frame in enumerate(frames_rgba):
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(frames_rgba) - 1:
                print(f"  Tightening frame {idx + 1}/{len(frames_rgba)}...")
            result_tight = apply_tightening(rgba_frame, seam_width=seam_width, threshold=threshold)
            frames_tight.append(result_tight)

        tightening_time = time.time() - tightening_start
        total_time = inference_time + tightening_time
        print(f"✓ Tightening completed in {tightening_time:.2f}s ({tightening_time/len(frames_rgba):.3f}s per frame)")
        print(f"✓ Total processing time: {total_time:.2f}s")

        return frames_rgba, frames_tight, durations

    else:
        # Single image processing
        start_time = time.time()

        image = img.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Generate mask
        print("Generating mask...")
        inference_start = time.time()

        # Reset GPU memory stats if using CUDA
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        inference_time = time.time() - inference_start

        # Get peak GPU memory usage
        if device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        # Convert mask to PIL and resize to original size
        mask = transforms.ToPILImage()(preds[0].squeeze())
        mask = mask.resize(image.size)

        # Apply mask to create transparent background
        result_rgba = image.copy()
        result_rgba.putalpha(mask)

        # Apply tightening to remove seams
        print("Applying tightening...")
        tightening_start = time.time()
        result_tight = apply_tightening(result_rgba, seam_width=seam_width, threshold=threshold)
        tightening_time = time.time() - tightening_start

        total_time = time.time() - start_time

        if device == 'cuda':
            print(f"✓ Inference: {inference_time:.2f}s, Tightening: {tightening_time:.2f}s, Total: {total_time:.2f}s")
            print(f"✓ Peak GPU memory: {peak_memory:.2f}GB")
        else:
            print(f"✓ Inference: {inference_time:.2f}s, Tightening: {tightening_time:.2f}s, Total: {total_time:.2f}s")

        return result_rgba, result_tight, None


def main():
    parser = argparse.ArgumentParser(
        description='Remove background from images using ToonOut (fine-tuned BiRefNet)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-download weights from HuggingFace
  python toonout_demo.py --input test.jpg

  # Use local weights file
  python toonout_demo.py --weights model.pth --input test.jpg
  python toonout_demo.py -w model.pth -i test.jpg -o result.png
        """
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to the fine-tuned BiRefNet weights (.pth file). If not provided, will auto-download from HuggingFace (joelseytre/toonout)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='sample.png',
        help='Path to the input image (default: sample.png)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save the output image (default: input_name_nobg.png)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for processing animated images (default: 8)'
    )

    parser.add_argument(
        '--seam-width', '-s',
        type=int,
        default=1,
        help='Seam width in pixels for tightening (default: 1)'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=200,
        help='Alpha threshold for tightening (0-255, default: 200)'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.weights and not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load model
    model = load_birefnet_with_custom_weights(args.weights)

    try:
        model.to(device)
        model.eval()

        # Process image
        result_rgba, result_tight, durations = remove_background(
            args.input, model, device, args.batch_size, args.seam_width, args.threshold
        )
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            print(f"\nWarning: CUDA error encountered: {str(e)[:100]}...")
            print("Falling back to CPU...\n")
            device = "cpu"
            model.to(device)
            model.eval()
            result_rgba, result_tight, durations = remove_background(
                args.input, model, device, args.batch_size, args.seam_width, args.threshold
            )
        else:
            raise

    # Determine output paths
    base_name = os.path.splitext(args.input)[0]
    ext = os.path.splitext(args.input)[1] or '.png'

    # Use .webp if input was animated
    if durations is not None:
        ext = '.webp'

    rgba_path = f"{base_name}.rgba{ext}"
    tight_path = f"{base_name}.rgba.tight{ext}"

    # Save results
    if durations is not None:
        # Save animated WebP
        result_rgba[0].save(
            rgba_path,
            save_all=True,
            append_images=result_rgba[1:],
            duration=durations,
            loop=0,
            lossless=True
        )
        result_tight[0].save(
            tight_path,
            save_all=True,
            append_images=result_tight[1:],
            duration=durations,
            loop=0,
            lossless=True
        )
        print(f"✓ Background removed successfully!")
        print(f"✓ Animated RGBA result saved to: {rgba_path}")
        print(f"✓ Animated tightened result saved to: {tight_path}")
    else:
        # Save single images
        result_rgba.save(rgba_path)
        result_tight.save(tight_path)
        print(f"✓ Background removed successfully!")
        print(f"✓ RGBA result saved to: {rgba_path}")
        print(f"✓ Tightened result saved to: {tight_path}")


if __name__ == "__main__":
    main()
