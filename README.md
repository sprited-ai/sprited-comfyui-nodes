# Sprited ComfyUI Nodes

A collection of custom nodes for ComfyUI

> [!NOTE]
> This projected was created with a [cookiecutter](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template. It helps you start writing custom nodes without worrying about the Python setup.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
4. Install dependencies: `pip install -r requirements.txt` or run `./install.sh`
5. Restart ComfyUI.

## Requirements

### Python Dependencies (installed automatically)
All dependencies are installed via pip - no system packages required!

- `requests>=2.25.0` - HTTP requests
- `opencv-python>=4.5.0` - Video processing
- `imageio>=2.25.0` - Video I/O
- `imageio-ffmpeg>=0.4.9` - **Bundles FFmpeg** (no system install needed!)
- `huggingface_hub>=0.20.0` - Model downloads
- `safetensors>=0.4.0` - Model loading

> **Note:** `imageio-ffmpeg` includes a pre-compiled FFmpeg binary, so you don't need to install FFmpeg on your system. This makes setup much easier across all platforms!

# Features

- **Video Downloader Node**: Download videos from URLs (YouTube, Vimeo, direct links, etc.) and get a preview with local file path output
- **Video Shot Splitter Node**: Automatically split videos into individual shots using scene detection algorithms

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd sprited_nodes
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name.
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
```

## Writing custom nodes

An example custom node is located in [node.py](src/sprited_nodes/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.
## Video Downloader Node Usage

The Video Downloader node allows you to download videos from direct URLs and provides the local file path.

### Inputs:
- **url** (STRING): The URL of the video to download (direct video URLs like `https://example.com/video.mp4`)
- **filename_prefix** (STRING): Prefix for the downloaded filename (default: "ComfyUI")

### Outputs:
- **file_path** (STRING): Local path to the downloaded video file

### Requirements:
The node only requires standard Python packages:
- `requests` for downloading

### Example Usage:
1. Add the "Video Downloader" node to your ComfyUI workflow
2. Enter a direct video URL: `https://example.com/path/myvideo.mp4`
3. Set a filename prefix like "MyProject" (or use default "ComfyUI")
4. The node will save the file as `MyProject_myvideo.mp4`
5. The node will download the video as-is and provide the file path
6. Use the file path output to connect to other video processing nodes

### Performance Notes:
- Downloads files directly using streaming HTTP requests
- Preserves original video format and quality
- Automatically extracts filename from URL and adds prefix
- Shows download progress for large files

### Notes:
- Downloads are stored in the system temp directory under `comfyui_video_downloads`
- Download timeout is set to 30 seconds
- The node will re-execute when the URL or filename_prefix changes

## Video Shot Splitter Node Usage

The Video Shot Splitter node automatically splits videos into individual shots using advanced scene detection algorithms.

### Inputs:
- **video** (STRING): Video file path or connect from other video nodes (VideoDownloader, LoadVideo, etc.)
- **detector** (CHOICE): Scene detection algorithm ("content" or "adaptive")
- **threshold** (FLOAT): Detection sensitivity threshold (0.1-50.0, default: 8.0)
- **min_scene_len** (INT): Minimum scene length in frames (1-300, default: 15)
- **output_format** (CHOICE): Output format for shot files ("mp4" or "webp")
- **reencode** (CHOICE): Whether to re-encode ("true" or "false", default: "false")
- **seconds_per_shot** (FLOAT, optional): Manual chunking interval in seconds (0 = auto detection)
- **output_dir** (STRING, optional): Output directory (empty = auto-generate)

### Outputs:
- **shot_file_paths** (STRING): Comma-separated list of paths to individual shot video files

### Detection Algorithms:
- **content**: Fast content-based detection (recommended for most videos)
- **adaptive**: Adaptive threshold detection (better for variable content)

### Example Usage:
1. Add the "Video Shot Splitter" node to your ComfyUI workflow
2. Connect a video input from another node (like Video Downloader) or enter a direct file path
3. Choose detection algorithm and adjust sensitivity if needed
4. The node will automatically detect scene changes and split the video
5. Output paths can be used to process individual shots separately

### Video Input Compatibility:
- Accepts STRING input (file paths)
- Compatible with outputs from VideoDownloader node
- Works with any node that outputs video file paths
- Automatically extracts file paths from various video object formats

### Manual Chunking:
- Set `seconds_per_shot` to a value > 0 to split video at fixed intervals
- Useful when you want uniform shot lengths instead of scene-based splitting
- Example: `seconds_per_shot = 5.0` creates 5-second clips

### Performance Notes:
- Fast copy mode (no re-encoding) for MP4 output when `reencode = false`
- Lossless re-encoding available for maximum compatibility
- WebP input files are automatically converted to temporary MP4 for processing
- Progress is shown in ComfyUI console during processing


- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile).
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

