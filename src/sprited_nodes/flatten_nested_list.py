# file: flatten_nested_list.py
#
# FlattenImageList – Flatten nested image lists
# -----------------------------------------------
#   Takes a list of image batches and flattens them into a single batch.
#
# Inputs:
#   • image_list: List of IMAGE batches
#
# Outputs:
#   • images: Single flattened IMAGE batch

import torch


class FlattenImageListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "flatten"
    CATEGORY = "image/batch"
    DESCRIPTION = "Flatten a list of image batches into a single batch."
    INPUT_IS_LIST = True

    def flatten(self, image_list):
        """
        Flatten nested image lists into a single batch.
        
        Args:
            image_list: List of IMAGE tensors (each with shape B, H, W, C)
            
        Returns:
            Single flattened IMAGE batch
        """
        if not image_list:
            raise ValueError("Image list is empty")
        
        # Flatten the list and concatenate all batches
        flattened = []
        for item in image_list:
            if isinstance(item, list):
                # If it's a nested list, flatten it
                for sub_item in item:
                    if sub_item is not None:
                        flattened.append(sub_item)
            elif item is not None:
                flattened.append(item)
        
        if not flattened:
            raise ValueError("No valid images found in the list")
        
        # Concatenate all batches along the batch dimension
        result = torch.cat(flattened, dim=0)
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FlattenImageList": FlattenImageListNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlattenImageList": "Flatten Image List",
}
