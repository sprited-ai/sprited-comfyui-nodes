from inspect import cleandoc
import os
import hashlib
import tempfile


class VideoDownloader:
    """
    A node that downloads videos from URLs
    
    This node takes a video URL, downloads the video file directly,
    and returns the local file path.
    """
    
    def __init__(self):
        # Create a downloads directory in the ComfyUI temp folder
        self.download_dir = os.path.join(tempfile.gettempdir(), "comfyui_video_downloads")
        os.makedirs(self.download_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the video downloader node.
        
        Returns:
            dict: Configuration for input fields
        """
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "https://example.com/video.mp4",
                    "tooltip": "URL of the video to download"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the downloaded filename"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "download_video"
    OUTPUT_NODE = False
    CATEGORY = "Video/Download"

    def download_video(self, url, filename_prefix):
        """
        Download video from URL and return file path
        
        Args:
            url (str): Video URL to download
            filename_prefix (str): Prefix for the filename
            
        Returns:
            tuple: (file_path,)
        """
        try:
            # Extract original filename from URL
            url_filename = os.path.basename(url.split('?')[0])
            if url_filename and '.' in url_filename:
                # Use original filename with prefix
                name_part, ext_part = os.path.splitext(url_filename)
                filename = f"{filename_prefix}_{name_part}{ext_part}"
            else:
                # Fallback to hash-based naming with .mp4 extension
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{filename_prefix}_video_{url_hash}.mp4"
            
            # Clean filename but preserve extension
            name_part, ext_part = os.path.splitext(filename)
            clean_name = "".join(c for c in name_part if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{clean_name}{ext_part}"
            
            output_path = os.path.join(self.download_dir, filename)
            
            # Download the file directly
            video_path = self._download_direct_video(url, output_path)
            
            if not video_path or not os.path.exists(video_path):
                raise Exception("Download failed - file not found")
            
            print(f"Video downloaded to: {video_path}")
            
            return (video_path,)
            
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return ("",)
    
    def _download_direct_video(self, url, output_path):
        """Download video directly using requests"""
        import requests
        
        try:
            print(f"Downloading video from: {url}")
            print(f"Saving to: {output_path}")
            
            # Stream download for large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
            
            print(f"\nDownload completed: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, url, filename_prefix):
        """
        Force re-execution when URL or filename_prefix changes
        """
        return hashlib.md5(f"{url}_{filename_prefix}".encode()).hexdigest()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoDownloader": VideoDownloader
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDownloader": "Video Downloader"
}
