from inspect import cleandoc

# Import the video downloader
from .download_video import VideoDownloader
from .split_shots import VideoShotSplitter
from .split_shots_v0 import VideoShotSplitterV0
from .split_shots_v2 import VideoShotSplitterV2
from .split_shots_v3 import VideoShotSplitterV3
from .extract_loop import LoopTrimNode
from .slice import SliceBatch, SliceLatents
from .url_to_video import URLToVideo
from .preview_video import PreviewVideo
from .split_shot_by_cut_score import ShotSplitByCutScore
from .pixel_stats import PixelRGBStats

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoDownloader": VideoDownloader,
    "VideoShotSplitter": VideoShotSplitter,
    "VideoShotSplitterV0": VideoShotSplitterV0,
    "VideoShotSplitterV2": VideoShotSplitterV2,
    "VideoShotSplitterV3": VideoShotSplitterV3,
    "ShotSplitByCutScore": ShotSplitByCutScore,
    "LoopTrimNode": LoopTrimNode,
    "SliceBatch": SliceBatch,
    "SliceLatents": SliceLatents,
    "URLToVideo": URLToVideo,
    "PreviewVideo": PreviewVideo,
    "PixelRGBStats": PixelRGBStats
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDownloader": "Video Downloader 🌱",
    "VideoShotSplitter": "Video Shot Splitter 🌱",
    "VideoShotSplitterV0": "Video Shot Splitter V0 (Simple) 🌱",
    "VideoShotSplitterV2": "Video Shot Splitter V2 🌱",
    "VideoShotSplitterV3": "Video Shot Splitter V3 (K-Shots) 🌱",
    "ShotSplitByCutScore": "Shot Split By Cut Score 🌱",
    "LoopTrimNode": "Loop Trim Node 🌱",
    "SliceBatch": "Image From Batch (Slice) 🌱",
    "SliceLatents": "Latent From Batch (Slice) 🌱",
    "URLToVideo": "URL to Video 🌱",
    "PreviewVideo": "Preview Video 🌱",
    "PixelRGBStats": "Pixel Stats (SpriteDX) 🌱"
}
