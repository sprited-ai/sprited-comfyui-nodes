import tempfile
from comfy.comfy_types.node_typing import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile

class PreviewVideo(ComfyNodeABC):
    """
    Takes any VIDEO and makes ComfyUI show its built-in preview player.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": (IO.VIDEO, {})}}
    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "preview"
    CATEGORY = "Preview"
    OUTPUT_NODE = True

    def preview(self, video):
        # 1) If it’s already a VideoFromFile, we’re done:
        if isinstance(video, VideoFromFile):
            return (video,)

        # 2) Otherwise write it out and wrap in VideoFromFile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        # video must implement .save_to(path)
        video.save_to(tmp.name)
        return (VideoFromFile(tmp.name),)
