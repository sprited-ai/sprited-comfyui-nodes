"""
BiRefNet Background Removal Node for ComfyUI
Remove background from images using fine-tuned BiRefNet model (ToonOut).
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from scipy.spatial import cKDTree
from inspect import cleandoc

# Simple fix for BiRefNet compatibility
import transformers.configuration_utils
original_getattribute = transformers.configuration_utils.PretrainedConfig.__getattribute__

def patched_getattribute(self, key):
    if key == 'is_encoder_decoder':
        return False
    return original_getattribute(self, key)

transformers.configuration_utils.PretrainedConfig.__getattribute__ = patched_getattribute

from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download


class BiRefNetBackgroundRemoval:
    """
    Remove background from images using fine-tuned BiRefNet model.
    
    This node uses the ToonOut fine-tuned BiRefNet model to generate
    high-quality background removal with optional seam tightening.
    """
    
    # Class variable to cache the model
    _model = None
    _device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "seam_width": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Width of seam removal for tightening (0 = no tightening)"
                }),
                "threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Alpha threshold for tightening (0-255)"
                }),
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Batch size for processing multiple images"
                }),
            },
            "optional": {
                "weights_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Optional: Path to custom weights file (leave empty to auto-download)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_images", "alpha_mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/segmentation"
    DESCRIPTION = cleandoc(__doc__)
    
    @classmethod
    def load_model(cls, checkpoint_path=None, device=None):
        """Load BiRefNet model with custom fine-tuned weights (cached)"""
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Return cached model if already loaded on correct device
        if cls._model is not None and cls._device == device:
            return cls._model, device
        
        print(f"Loading BiRefNet model on {device}...")
        
        # Load the base model from HuggingFace
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        
        # Download weights from HuggingFace if no local path provided
        if checkpoint_path is None or checkpoint_path == "":
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
        model.to(device)
        model.eval()
        
        # Cache the model
        cls._model = model
        cls._device = device
        
        print("BiRefNet model loaded successfully!")
        return model, device
    
    @staticmethod
    def apply_tightening(image_pil, seam_width=1, threshold=200):
        """
        Apply seam removal to background-matted image.
        
        Args:
            image_pil: PIL Image in RGBA format
            seam_width: Width of seam to remove
            threshold: Alpha threshold (0-255)
            
        Returns:
            PIL Image with tightened alpha
        """
        if seam_width == 0:
            return image_pil
        
        # Split into RGB and alpha
        rgb = image_pil.convert('RGB')
        alpha = image_pil.getchannel('A')
        
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
    
    def remove_background(self, images, seam_width=1, threshold=200, batch_size=4, weights_path=""):
        """
        Remove background from images using BiRefNet.
        
        Args:
            images: ComfyUI IMAGE tensor (B, H, W, C) in RGB format, values 0-1
            seam_width: Width of seam removal
            threshold: Alpha threshold
            batch_size: Batch size for processing
            weights_path: Optional path to custom weights
            
        Returns:
            tuple: (rgba_images, alpha_mask)
        """
        # Load model (cached)
        model, device = self.load_model(
            checkpoint_path=weights_path if weights_path else None,
            device=None
        )
        
        # Image preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        num_images = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        
        print(f"Processing {num_images} images with BiRefNet...")
        
        rgba_results = []
        alpha_masks = []
        
        # Process in batches
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_images = images[batch_start:batch_end]
            
            print(f"  Processing images {batch_start + 1}-{batch_end}/{num_images}...")
            
            # Convert ComfyUI tensors to PIL images and prepare batch
            pil_images = []
            batch_tensors = []
            
            for i in range(batch_images.shape[0]):
                # Convert from ComfyUI format (0-1, RGB) to PIL
                img_np = (batch_images[i].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='RGB')
                pil_images.append(pil_img)
                
                # Prepare for model
                batch_tensors.append(transform(pil_img))
            
            # Stack into batch and move to device
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Generate masks
            with torch.no_grad():
                preds = model(batch_tensor)[-1].sigmoid().cpu()
            
            # Process each image in batch
            for i, (pil_img, pred) in enumerate(zip(pil_images, preds)):
                # Convert mask to PIL and resize to original size
                mask = transforms.ToPILImage()(pred.squeeze())
                mask = mask.resize((w, h))
                
                # Apply mask to create transparent background
                result_rgba = pil_img.copy()
                result_rgba = result_rgba.resize((w, h))
                result_rgba.putalpha(mask)
                
                # Apply tightening if requested
                if seam_width > 0:
                    result_rgba = self.apply_tightening(result_rgba, seam_width=seam_width, threshold=threshold)
                
                # Convert back to ComfyUI format
                # RGBA image
                rgba_np = np.array(result_rgba).astype(np.float32) / 255.0
                rgba_tensor = torch.from_numpy(rgba_np)
                rgba_results.append(rgba_tensor)
                
                # Alpha mask only
                alpha_np = np.array(result_rgba.getchannel('A')).astype(np.float32) / 255.0
                alpha_tensor = torch.from_numpy(alpha_np)
                alpha_masks.append(alpha_tensor)
        
        # Stack results
        rgba_batch = torch.stack(rgba_results)  # (B, H, W, 4)
        alpha_batch = torch.stack(alpha_masks)  # (B, H, W)
        
        print(f"✓ Background removal completed for {num_images} images")
        
        return (rgba_batch, alpha_batch)


# Node registration
NODE_CLASS_MAPPINGS = {
    "BiRefNetBackgroundRemoval": BiRefNetBackgroundRemoval,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNetBackgroundRemoval": "BiRefNet Background Removal (ToonOut) 🌱",
}
