#!/bin/bash
# install.sh - Installation script for Sprited ComfyUI Nodes

echo "Installing Sprited ComfyUI Nodes dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies (including bundled FFmpeg)..."
pip install -r requirements.txt

echo ""
echo "🎉 Installation complete!"
echo ""
echo "ℹ️  Note: FFmpeg is included via imageio-ffmpeg package - no system install needed!"
echo ""
echo "The Sprited ComfyUI Nodes should now be available in ComfyUI."
echo ""
echo "Available nodes:"
echo "  - Video Downloader: Download videos from URLs"
echo "  - Video Even Shot Splitter: Split videos into equal-length shots"
echo "  - Loop Extractor V2: Extract seamless loops from videos"
echo "  - Sprite Anti-Corruption V1: Restore corrupted sprite animations"
echo "  - Pixel RGB Stats: Analyze pixel statistics with masking"
