#!/usr/bin/env python3
"""
Simple test script for the VideoDownloader node
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sprited_nodes.download_video import VideoDownloader

def test_video_downloader():
    """Test the VideoDownloader node with a sample video"""
    
    # Create the node
    downloader = VideoDownloader()
    
    # Test with a direct video URL (small sample video)
    test_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
    
    try:
        print(f"Testing direct video download from: {test_url}")
        
        # Download the video
        file_path, = downloader.download_video(
            url=test_url,
            filename_prefix="TestDownload"
        )
        
        if file_path and os.path.exists(file_path):
            print(f"✅ Video downloaded successfully to: {file_path}")
            print(f"File size: {os.path.getsize(file_path)} bytes")
        else:
            print("❌ Video download failed")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")

if __name__ == "__main__":
    test_video_downloader()
