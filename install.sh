#!/bin/bash
# install.sh - Installation script for Sprited ComfyUI Nodes

echo "Installing Sprited ComfyUI Nodes dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is already installed"
else
    echo "⚠️  FFmpeg is not installed. Please install it:"
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
fi

# Check for webp tools (optional)
if command -v webpmux &> /dev/null; then
    echo "✓ WebP tools are already installed"
else
    echo "ℹ️  WebP tools not found (optional for WebP support):"
    echo "   Ubuntu/Debian: sudo apt install webp"
    echo "   macOS: brew install webp"
fi

echo ""
echo "🎉 Installation complete!"
echo "The Sprited ComfyUI Nodes should now be available in ComfyUI."
echo ""
echo "Available nodes:"
echo "  - Video Downloader: Download videos from URLs"
echo "  - Video Shot Splitter: Split videos into individual shots"
