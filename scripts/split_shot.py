from PIL import Image, ImageSequence, ImageDraw
import numpy as np
from scipy.signal import find_peaks
import fire
import os


# ---------------------------------------------------------------------
# Scene-cut detection
# ---------------------------------------------------------------------
def detect_cuts_visual(
    images,
    k=3,
    downscale=64,
    prominence_ratio=0.25,
):
    """Detect k-1 scene cuts using downscaled grayscale frame differences."""
    if not images or len(images) < 2:
        return [], []

    grays = [
        np.array(img.convert("L").resize((downscale, downscale)), dtype=np.float32)
        for img in images
    ]

    diffs = np.array(
        [np.mean(np.abs(grays[i - 1] - grays[i])) for i in range(1, len(grays))]
    )

    diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min() + 1e-8)

    peaks, _ = find_peaks(diffs, prominence=prominence_ratio * diffs.max())
    if len(peaks) > k - 1:
        top_idx = np.argsort(diffs[peaks])[-(k - 1) :]
        peaks = sorted(peaks[top_idx])

    return list(enumerate(diffs)), peaks


# ---------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------
def load_animation_frames(path):
    im = Image.open(path)
    frames, durations = [], []
    for frame in ImageSequence.Iterator(im):
        frames.append(frame.copy().convert("RGB"))
        durations.append(frame.info.get("duration", im.info.get("duration", 100)))
    return frames, durations


def save_segments(frames, durations, cuts, out_prefix):
    segments = []
    indices = [0] + cuts + [len(frames)]
    outputs_dir = os.path.join("outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    for i in range(len(indices) - 1):
        start, end = indices[i], indices[i + 1]
        seg_frames = frames[start:end]
        seg_durations = durations[start:end]
        out_path = os.path.join(outputs_dir, f"{os.path.basename(out_prefix)}_part{i:02d}.webp")
        seg_frames[0].save(
            out_path,
            format="WEBP",
            save_all=True,
            append_images=seg_frames[1:],
            duration=seg_durations,
            loop=0,
        )
        segments.append(out_path)
        print(f"Saved segment {i}: frames {start}-{end-1} ({len(seg_frames)} frames)")
    return segments


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def visualize_cuts(frames, diffs, cuts, thumb_size=128, bar_max_width=200, out_path="diagnostic.png"):
    """Create a tall PNG visualizing each frame and its diff score as a bar."""
    frame_count = len(frames)
    chunk_size = 64
    num_chunks = (frame_count + chunk_size - 1) // chunk_size
    diffs_values = np.array([d for _, d in diffs])

    # Precompute grayscale downscaled frames for diffs
    downscale = thumb_size
    grays = [np.array(img.convert("L").resize((downscale, downscale)), dtype=np.float32) for img in frames]

    outputs_dir = os.path.join("outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, frame_count)
        chunk_frames = frames[start:end]
        chunk_diffs = diffs[start:end]
        width = thumb_size * 2 + bar_max_width + 70  # extra space for diff image
        height = len(chunk_frames) * thumb_size
        img = Image.new("RGB", (width, height), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        for i, (idx, diff) in enumerate(chunk_diffs):
            y = i * thumb_size
            # Draw frame thumbnail
            thumb = frames[idx - 1].resize((thumb_size, thumb_size))
            img.paste(thumb, (0, y))

            # Draw visual diff (grayscale, white=change, black=no change)
            if idx > 1:
                diff_img = np.abs(grays[idx - 1] - grays[idx - 2])
                diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min() + 1e-8)
                diff_img = (diff_img * 255).astype(np.uint8)
                diff_pil = Image.fromarray(diff_img, mode="L").convert("RGB")
            else:
                diff_pil = Image.new("RGB", (thumb_size, thumb_size), (0, 0, 0))
            img.paste(diff_pil, (thumb_size + 5, y - thumb_size))

            # Draw score bar
            bar_x = thumb_size * 2 + 10
            bar_len = int(diff * bar_max_width)
            color = (255, 80, 80) if idx in cuts else (80, 200, 255)
            draw.rectangle(
                [bar_x, y + 10, bar_x + bar_len, y + thumb_size - 10],
                fill=color,
            )

            # Draw frame index text
            draw.text((thumb_size * 2 + bar_max_width + 15, y + thumb_size // 3),
                      f"{idx:03d} ({diff:.2f})",
                      fill=(255, 255, 255))

        if num_chunks == 1:
            out_file = os.path.join(outputs_dir, os.path.basename(out_path))
        else:
            base, ext = os.path.splitext(os.path.basename(out_path))
            out_file = os.path.join(outputs_dir, f"{base}_part{chunk_idx:02d}{ext}")
        img.save(out_file)
        print(f"Saved visualization to {out_file} (size: {img.width}x{img.height})")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main(
    path,
    k=3,
    downscale=64,
    prominence_ratio=0.25,
    out_prefix=None,
    visualize=True,
):
    """Split an animated .webp into k segments, preserving frame rate and generating a vertical diagnostic PNG."""
    out_prefix = out_prefix or os.path.splitext(os.path.basename(path))[0]
    print(f"Loading frames from {path}...")
    frames, durations = load_animation_frames(path)
    print(f"Loaded {len(frames)} frames.\n")

    diffs, cuts = detect_cuts_visual(
        frames,
        k=k,
        downscale=downscale,
        prominence_ratio=prominence_ratio,
    )

    print("Frame-to-frame diff scores:")
    for idx, diff in diffs:
        mark = " <-- CUT" if idx in cuts else ""
        print(f"Frame {idx:03d}: diff = {diff:5.3f}{mark}")

    print(f"\nSelected {len(cuts)} cuts (k={k} → {k-1} cuts): {cuts}\n")
    save_segments(frames, durations, cuts, out_prefix)

    if visualize:
        viz_path = f"{out_prefix}_diagnostic.png"
        visualize_cuts(frames, diffs, cuts, thumb_size=128, out_path=viz_path)


if __name__ == "__main__":
    fire.Fire(main)
