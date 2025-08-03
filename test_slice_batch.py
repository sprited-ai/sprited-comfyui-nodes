#!/usr/bin/env python3
"""
Simple test script for the SliceBatch nodes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from sprited_nodes.slice import SliceBatch, SliceLatents

def test_slice_batch_image():
    """Test SliceBatch with IMAGE tensor input (following ComfyUI ImageFromBatch pattern)"""
    
    # Create the node
    slicer = SliceBatch()
    
    # Create a test IMAGE tensor (batch, height, width, channels)
    test_images = torch.randn(8, 512, 512, 3)  # 8 images in batch
    
    try:
        print("Testing SliceBatch with IMAGE input...")
        
        # Test slicing: start at index 2, take 3 images
        result, = slicer.slice(image=test_images, batch_index=2, length=3)
        
        print(f"✅ IMAGE slicing successful!")
        print(f"Original shape: {test_images.shape}")
        print(f"Sliced shape: {result.shape}")
        print(f"Expected: torch.Size([3, 512, 512, 3])")
        
        assert result.shape[0] == 3, f"Expected batch size 3, got {result.shape[0]}"
        assert torch.equal(result, test_images[2:5]), "Sliced content doesn't match expected"
        
    except Exception as e:
        print(f"❌ Error during IMAGE test: {str(e)}")
        return False
        
    return True

def test_slice_latents():
    """Test SliceLatents with LATENT dict input (following ComfyUI LATENT patterns)"""
    
    # Create the node
    slicer = SliceLatents()
    
    # Create a test LATENT dict (ComfyUI standard format)
    test_latent = {
        "samples": torch.randn(5, 4, 64, 64),  # 5 latents in batch
        "noise_mask": torch.ones(5, 1, 64, 64)  # optional noise mask
    }
    
    try:
        print("Testing SliceLatents with LATENT input...")
        
        # Test slicing: start at index 1, take 2 latents
        result, = slicer.slice(samples=test_latent, batch_index=1, length=2)
        
        print(f"✅ LATENT slicing successful!")
        print(f"Original samples shape: {test_latent['samples'].shape}")
        print(f"Sliced samples shape: {result['samples'].shape}")
        print(f"Expected: torch.Size([2, 4, 64, 64])")
        
        assert result['samples'].shape[0] == 2, f"Expected batch size 2, got {result['samples'].shape[0]}"
        assert torch.equal(result['samples'], test_latent['samples'][1:3]), "Sliced samples don't match expected"
        
        # Check noise_mask was also sliced
        if 'noise_mask' in result:
            assert result['noise_mask'].shape[0] == 2, f"Expected noise_mask batch size 2, got {result['noise_mask'].shape[0]}"
            assert torch.equal(result['noise_mask'], test_latent['noise_mask'][1:3]), "Sliced noise_mask doesn't match expected"
        
    except Exception as e:
        print(f"❌ Error during LATENT test: {str(e)}")
        return False
        
    return True

def test_bounds_checking():
    """Test that bounds checking works correctly"""
    
    # Create the nodes
    image_slicer = SliceBatch()
    latent_slicer = SliceLatents()
    
    # Test IMAGE bounds checking
    test_images = torch.randn(3, 256, 256, 3)  # Only 3 images
    
    try:
        print("Testing IMAGE bounds checking...")
        
        # Test out-of-bounds slicing (should clamp to available)
        result, = image_slicer.slice(image=test_images, batch_index=1, length=10)  
        
        print(f"✅ IMAGE bounds checking successful!")
        print(f"Original shape: {test_images.shape}")
        print(f"Sliced shape: {result.shape}")
        print(f"Expected: torch.Size([2, 256, 256, 3]) (clamped to available)")
        
        assert result.shape[0] == 2, f"Expected batch size 2, got {result.shape[0]}"
        
    except Exception as e:
        print(f"❌ Error during IMAGE bounds test: {str(e)}")
        return False
    
    # Test LATENT bounds checking
    test_latent = {"samples": torch.randn(2, 4, 32, 32)}  # Only 2 latents
    
    try:
        print("Testing LATENT bounds checking...")
        
        # Test out-of-bounds slicing
        result, = latent_slicer.slice(samples=test_latent, batch_index=0, length=5)
        
        print(f"✅ LATENT bounds checking successful!")
        print(f"Original samples shape: {test_latent['samples'].shape}")
        print(f"Sliced samples shape: {result['samples'].shape}")
        print(f"Expected: torch.Size([2, 4, 32, 32]) (clamped to available)")
        
        assert result['samples'].shape[0] == 2, f"Expected batch size 2, got {result['samples'].shape[0]}"
        
    except Exception as e:
        print(f"❌ Error during LATENT bounds test: {str(e)}")
        return False
        
    return True

def test_input_types():
    """Test that INPUT_TYPES returns the expected structure"""
    
    # Test SliceBatch
    input_types = SliceBatch.INPUT_TYPES()
    required = input_types.get('required', {})
    expected_required = ['image', 'batch_index', 'length']
    
    for field in expected_required:
        if field not in required:
            print(f"✗ SliceBatch missing required field: {field}")
            return False
    
    # Test SliceLatents
    input_types = SliceLatents.INPUT_TYPES()
    required = input_types.get('required', {})
    expected_required = ['samples', 'batch_index', 'length']
    
    for field in expected_required:
        if field not in required:
            print(f"✗ SliceLatents missing required field: {field}")
            return False
    
    print("✓ INPUT_TYPES structures are correct")
    return True

def test_node_attributes():
    """Test that the nodes have all required ComfyUI attributes"""
    
    # Check class attributes for both nodes
    nodes_to_test = [
        (SliceBatch, "SliceBatch"),
        (SliceLatents, "SliceLatents")
    ]
    
    attributes = ['RETURN_TYPES', 'FUNCTION', 'CATEGORY', 'DESCRIPTION']
    
    for node_class, node_name in nodes_to_test:
        for attr in attributes:
            if not hasattr(node_class, attr):
                print(f"✗ {node_name} missing attribute: {attr}")
                return False
        
        # Check that the function method exists
        if not hasattr(node_class(), node_class.FUNCTION):
            print(f"✗ {node_name} method {node_class.FUNCTION} not found")
            return False
    
    print("✓ All required attributes present for both nodes")
    return True

def main():
    print("Testing SliceBatch & SliceLatents ComfyUI Nodes...")
    print("=" * 50)
    print()
    
    # Run tests
    tests = [
        test_input_types,
        test_node_attributes,
        test_slice_batch_image,
        test_slice_latents,
        test_bounds_checking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The slice nodes should work correctly in ComfyUI.")
        print("\nNodes follow ComfyUI conventions:")
        print("- SliceBatch: similar to 'Image From Batch' but with length parameter")  
        print("- SliceLatents: similar to 'Latent From Batch' but with length parameter")
        print("- Both use 'batch_index' parameter name like other ComfyUI batch nodes")
        print("- Both handle bounds checking automatically")
        print("- Both preserve ComfyUI data formats (IMAGE tensors, LATENT dicts)")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
