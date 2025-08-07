from inspect import cleandoc

# Import the video downloader
from .download_video import VideoDownloader
from .split_shots import VideoShotSplitter
from .extract_loop import LoopTrimNode
from .slice import SliceBatch, SliceLatents
from .url_to_video import URLToVideo
from .preview_video import PreviewVideo

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoDownloader": VideoDownloader,
    "VideoShotSplitter": VideoShotSplitter,
    "LoopTrimNode": LoopTrimNode,
    "SliceBatch": SliceBatch,
    "SliceLatents": SliceLatents,
    "URLToVideo": URLToVideo,
    "PreviewVideo": PreviewVideo
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDownloader": "Video Downloader 🌱",
    "VideoShotSplitter": "Video Shot Splitter 🌱",
    "LoopTrimNode": "Loop Trim Node 🌱",
    "SliceBatch": "Image From Batch (Slice) 🌱",
    "SliceLatents": "Latent From Batch (Slice) 🌱",
    "URLToVideo": "URL to Video 🌱",
    "PreviewVideo": "Preview Video 🌱"
}
