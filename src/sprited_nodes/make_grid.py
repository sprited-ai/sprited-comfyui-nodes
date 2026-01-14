# file: make_grid.py
#
# MakeGrid – Pack image batch into a grid
# ----------------------------------------
#   Takes an IMAGE batch and arranges frames into a grid layout
#   with a maximum width constraint.
#
# Inputs:
#   • images: IMAGE batch to pack
#   • width: Maximum width in pixels for the grid
#   • padding: Pixels of padding between images
#
# Outputs:
#   • grid_image: Packed grid as IMAGE

import torch
import numpy as np
import math


class MakeGridNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 2048, "min": 1}),
                "padding": ("INT", {"default": 0, "min": 0}),
                "min_height": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)
    FUNCTION = "make_grid"
    CATEGORY = "image/batch"
    DESCRIPTION = "Arrange image batch into a grid with width constraint."

    def make_grid(self, images, width=2048, padding=0, min_height=0):
        """
        Pack images into a grid layout.
        
        Args:
            images: ComfyUI IMAGE tensor (B, H, W, C) in RGB format, values 0-1
            width: Maximum width of the grid in pixels
            padding: Pixels of padding between images
            min_height: Minimum height of the grid in pixels
            
        Returns:
            grid_image: Packed grid as IMAGE tensor
        """
        batch_size = images.shape[0]
        if batch_size == 0:
            raise ValueError("Image batch is empty")
        
        # Get dimensions of a single frame
        frame_height = images.shape[1]
        frame_width = images.shape[2]
        channels = images.shape[3]
        
        # Calculate how many columns fit in width
        frame_width_padded = frame_width + padding
        cols = max(1, (width + padding) // frame_width_padded)
        
        # Calculate number of rows needed
        rows = math.ceil(batch_size / cols)
        
        # Calculate actual grid dimensions
        grid_width = cols * frame_width + (cols - 1) * padding
        grid_height = rows * frame_height + (rows - 1) * padding
        
        # Ensure minimum height is met
        if min_height > 0:
            grid_height = max(grid_height, min_height)
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width, channels), dtype=np.float32)
        
        # Place each frame in the grid
        for idx in range(batch_size):
            row = idx // cols
            col = idx % cols
            
            y = row * (frame_height + padding)
            x = col * (frame_width + padding)
            
            frame = images[idx].cpu().numpy()
            grid[y:y+frame_height, x:x+frame_width] = frame
        
        # Convert to tensor and add batch dimension
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)  # (1, H, W, C)
        
        return (grid_tensor,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MakeGrid": MakeGridNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MakeGrid": "Make Grid",
}
