import os
import tempfile
import requests
from pathlib import Path

# ComfyUI video types - following the pattern from your other nodes
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    # Fallback if ComfyUI API is not available
    VideoFromFile = str

class URLToVideo:
    """
    Download a video from a URL and emit it as a VideoFromFile node.
    Uses ComfyUI's temp directory and proper VIDEO type handling.
    """
    
    def __init__(self):
        # Use ComfyUI's temp directory pattern
        self.temp_dir = Path(tempfile.gettempdir()) / "comfyui_url_videos"
        self.temp_dir.mkdir(exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL of the video to download"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"
    CATEGORY = "video/download"
    DESCRIPTION = "Download a video from URL and return as VIDEO type for ComfyUI processing."

    def load_video(self, url):
        if not url.strip():
            raise ValueError("URL cannot be empty")
        
        # Extract filename from URL, fallback to hash-based naming
        try:
            url_path = Path(url.split('?')[0])  # Remove query params
            if url_path.suffix:
                ext = url_path.suffix
                base_name = url_path.stem
            else:
                ext = ".mp4"
                base_name = "video"
        except:
            ext = ".mp4"
            base_name = "video"
        
        # Create unique filename to avoid conflicts
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{base_name}_{url_hash}{ext}"
        output_path = self.temp_dir / filename
        
        # Download if not already cached
        if not output_path.exists():
            try:
                print(f"Downloading video from: {url}")
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            
                print(f"Video downloaded to: {output_path}")
                
            except Exception as e:
                # Clean up partial download
                if output_path.exists():
                    output_path.unlink()
                raise ValueError(f"Failed to download video: {str(e)}")
        else:
            print(f"Using cached video: {output_path}")

        # Return proper ComfyUI VIDEO type
        return (VideoFromFile(str(output_path)),)
