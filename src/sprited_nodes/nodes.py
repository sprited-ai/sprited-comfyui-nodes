from inspect import cleandoc

# Import the video downloader
from .download_video import VideoDownloader
from .slice import SliceBatch, SliceLatents
from .url_to_video import URLToVideo
from .pixel_stats import PixelRGBStats
from .even_shot_splitter import VideoEvenShotSplitter
from .extract_loop_v2 import LoopExtractorNodeV2
from .extract_loop_v3 import LoopExtractorNodeV3
from .anti_corruption import SpriteDXAntiCorruptionV1
from .make_grid import MakeGridNode
from .flatten_nested_list import FlattenImageListNode
from .parse_int import SpriteDXParseIntNode
from .birefnet_background_removal import BiRefNetBackgroundRemoval


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoDownloader": VideoDownloader,
    "VideoEvenShotSplitter": VideoEvenShotSplitter,
    "LoopExtractorNodeV2": LoopExtractorNodeV2,
    "LoopExtractorNodeV3": LoopExtractorNodeV3,
    "SliceBatch": SliceBatch,
    "SliceLatents": SliceLatents,
    "URLToVideo": URLToVideo,
    "PixelRGBStats": PixelRGBStats,
    "SpriteDXAntiCorruptionV1": SpriteDXAntiCorruptionV1,
    "SpritedMakeGrid": MakeGridNode,
    "FlattenImageList": FlattenImageListNode,
    "SpriteDX_ParseInt": SpriteDXParseIntNode,
    "BiRefNetBackgroundRemoval": BiRefNetBackgroundRemoval
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDownloader": "Video Downloader 🌱",
    "VideoEvenShotSplitter": "Video Even Shot Splitter 🌱",
    "LoopExtractorNodeV2": "Loop Extractor Node V2 🌱",
    "LoopExtractorNodeV3": "Loop Extractor Node V3 🌱",
    "SliceBatch": "Image From Batch (Slice) 🌱",
    "SliceLatents": "Latent From Batch (Slice) 🌱",
    "URLToVideo": "URL to Video 🌱",
    "PixelRGBStats": "Pixel Stats (SpriteDX) 🌱",
    "SpriteDXAntiCorruptionV1": "SpriteDX Anti-Corruption V1 🌱",
    "SpritedMakeGrid": "Make Grid 🌱",
    "FlattenImageList": "Flatten Image List 🌱",
    "SpriteDX_ParseInt": "Parse Int (SpriteDX) 🌱",
    "BiRefNetBackgroundRemoval": "BiRefNet Background Removal (ToonOut) 🌱"
}
