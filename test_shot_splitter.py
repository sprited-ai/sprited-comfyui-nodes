#!/usr/bin/env python3
"""
Test script for the VideoShotSplitter ComfyUI node
"""

from src.sprited_nodes.split_shots import VideoShotSplitter

def test_node_creation():
    """Test that the node can be created successfully"""
    node = VideoShotSplitter()
    print("✓ Node created successfully")
    
def test_input_types():
    """Test that INPUT_TYPES returns the expected structure"""
    input_types = VideoShotSplitter.INPUT_TYPES()
    
    # Check that required fields exist
    required = input_types.get('required', {})
    expected_required = ['file_path', 'detector', 'threshold', 'min_scene_len', 'output_format', 'reencode']
    
    for field in expected_required:
        if field not in required:
            print(f"✗ Missing required field: {field}")
            return False
    
    # Check that optional fields exist
    optional = input_types.get('optional', {})
    expected_optional = ['seconds_per_shot', 'output_dir']
    
    for field in expected_optional:
        if field not in optional:
            print(f"✗ Missing optional field: {field}")
            return False
    
    print("✓ INPUT_TYPES structure is correct")
    return True

def test_node_attributes():
    """Test that the node has all required ComfyUI attributes"""
    node = VideoShotSplitter()
    
    # Check class attributes
    attributes = [
        'RETURN_TYPES', 'RETURN_NAMES', 'DESCRIPTION', 
        'FUNCTION', 'OUTPUT_NODE', 'CATEGORY'
    ]
    
    for attr in attributes:
        if not hasattr(VideoShotSplitter, attr):
            print(f"✗ Missing attribute: {attr}")
            return False
    
    # Check that the function method exists
    if not hasattr(node, VideoShotSplitter.FUNCTION):
        print(f"✗ Method {VideoShotSplitter.FUNCTION} not found")
        return False
    
    print("✓ All required attributes present")
    return True

def main():
    print("Testing VideoShotSplitter ComfyUI Node...")
    print()
    
    # Run tests
    tests = [
        test_node_creation,
        test_input_types,
        test_node_attributes
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
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The node is ready for ComfyUI.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
