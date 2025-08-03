# file: custom_nodes/slice_batch.py
#
# SliceBatch – trim IMAGE or LATENT batches
# ----------------------------------------
#   • `start`  – first index to keep  (0-based)
#   • `length` – how many items you want
#
# Example: start=0 length=3 → keep the first 3 frames/latents.

import torch

class SliceBatch:
    """Slice a batch of images - similar to ComfyUI's ImageFromBatch but with length parameter"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "slice"
    CATEGORY = "image/batch"
    DESCRIPTION = "Extract a contiguous slice from an IMAGE batch starting at batch_index with specified length."

    def slice(self, image, batch_index: int, length: int):
        """
        Extract a slice from IMAGE batch.
        Based on ComfyUI's ImageFromBatch pattern.
        """
        s_in = image
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return (s,)


class SliceLatents:
    """Slice a batch of latents - following ComfyUI LATENT batch patterns"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "samples": ("LATENT",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "slice"
    CATEGORY = "latent/batch"
    DESCRIPTION = "Extract a contiguous slice from a LATENT batch starting at batch_index with specified length."

    def slice(self, samples, batch_index: int, length: int):
        """
        Extract a slice from LATENT batch.
        Based on ComfyUI's LATENT batch handling patterns.
        """
        s = samples.copy()
        s_in = s["samples"]
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s["samples"] = s_in[batch_index:batch_index + length].clone()
        
        # Handle noise_mask if present (following ComfyUI patterns)
        if "noise_mask" in s:
            mask = s["noise_mask"]
            if mask is not None:
                s["noise_mask"] = mask[batch_index:batch_index + length].clone()
        
        return (s,)
