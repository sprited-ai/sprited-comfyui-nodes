import os, tempfile
from datetime import datetime
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    VideoFromFile = str

class LoopExtractorNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "min_len": ("INT", {"default": 4, "min": 1, "max": 40}),
                "max_len": ("INT", {"default": 40, "min": 4, "max": 120}),
            },
            "optional": {
                "output_format": (["webp", "mp4"], {"default": "webp"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    OUTPUT_IS_LIST = (False,)
    RETURN_NAMES = ("loop_video",)
    FUNCTION = "extract_loop"
    CATEGORY = "Video/Edit"
    DESCRIPTION = "Extract best seamless loop from video using composite frame similarity and length penalty."

    def extract_loop(self, video, min_len=4, max_len=40, output_format="webp", overwrite=True):
        import cv2
        from PIL import Image
        import numpy as np
        from pathlib import Path
        # Helper: extract frames from webp/mp4
        def extract_frames(path):
            frames = []
            ext = str(path).lower()
            if ext.endswith(".webp"):
                with Image.open(path) as im:
                    try:
                        while True:
                            frame = im.convert('RGB')
                            frames.append(np.array(frame)[:, :, ::-1])  # RGB to BGR
                            im.seek(im.tell() + 1)
                    except EOFError:
                        pass
            else:
                cap = cv2.VideoCapture(str(path))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
            return frames

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def evaluate_length_penalty(length, mag=0.0045):
            x = length
            y = mag + -mag * sigmoid(0.58 * (x - 1.65)) + 0.25 * mag * sigmoid(2 * (x - 17))
            return y

        def compute_sim(f1, f2):
            v1 = f1.flatten()
            v2 = f2.flatten()
            eps = 1e-8
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
            return cos_sim

        def detect_loops(frames, min_len=4, max_len=40, top_k=1):
            n = len(frames)
            candidates = []
            processed_frames = [
                cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
                for f in frames
            ]
            composite_frames = []
            motion_energies = []
            for idx in range(n):
                prev_idx = (idx - 1) % n
                next_idx = (idx + 1) % n
                r = processed_frames[prev_idx]
                g = processed_frames[idx]
                b = processed_frames[next_idx]
                composite = np.stack([r, g, b], axis=-1)
                composite_frames.append(composite)
                if idx == n - 1:
                    motion_energies.append(0.0)
                else:
                    motion_energy = np.mean(np.abs(processed_frames[next_idx] - processed_frames[idx]))
                    motion_energies.append(motion_energy)
            max_motion_energy = 20.0
            normalized_motion_energies = [min(me / max_motion_energy, 1.0) for me in motion_energies]
            for i in range(n):
                for j in range(i+min_len, min(i+max_len, n)):
                    start_comp = composite_frames[i].astype(np.float32)
                    end_comp = composite_frames[j].astype(np.float32)
                    cos_sim = compute_sim(start_comp, end_comp)
                    length_penalty = evaluate_length_penalty(j - i)
                    nme = normalized_motion_energies[i]
                    similarity_correction = (1.0 - cos_sim) * nme * 0.5
                    score = cos_sim + similarity_correction - length_penalty
                    candidates.append((score, i, j, cos_sim, length_penalty))
            candidates.sort(reverse=True)
            return candidates[:top_k]

        # Main logic
        src_path = video if isinstance(video, str) else getattr(video, "_VideoFromFile__file", None)
        if not src_path or not os.path.exists(src_path):
            raise RuntimeError("Unable to resolve video file path.")
        frames = extract_frames(src_path)
        if len(frames) < max_len:
            raise RuntimeError(f"Not enough frames ({len(frames)}) for loop extraction.")
        best = detect_loops(frames, min_len, max_len, top_k=1)[0]
        score, i, j, cos_sim, length_penalty = best
        loop_frames = frames[i:j]
        # Save as animated webp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(tempfile.mkdtemp(prefix="loopextract_"))
        out_name = f"loop_{timestamp}.webp"
        out_path = out_dir / out_name
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in loop_frames]
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=40,
            loop=0,
            lossless=True
        )
        return (VideoFromFile(str(out_path)),)
"""
Inspiration:

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json

shots_dir = Path('data/shots')
files = sorted(shots_dir.glob('sample-*.webp'))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate_length_penalty(len, mag = 0.0045):
    # Length penalty function based on sigmoid curves.
    x = len
    y = mag + -mag * sigmoid(0.58 * (x - 1.65)) + 0.25 * mag * sigmoid(2 * (x - 17))
    return y

# Two sigmoids — one dips around x=2, one rises around x=16
import matplotlib.pyplot as plt
x = np.linspace(0, 40, 400)
y = evaluate_length_penalty(x)
plt.plot(x, y, 'k', lw=3)
plt.axvline(4, color='k', lw=2)
plt.axvline(16, color='k', lw=2)
plt.text(0, -0.06, '0', ha='center', va='top', fontsize=12)
plt.text(2, -0.06, '2', ha='center', va='top', fontsize=12)
plt.text(16, -0.06, '16', ha='center', va='top', fontsize=12)
# save the plot into length_penalty_plot.png
plt.savefig('length_penalty_plot.png')
plt.close()

def extract_frames(webp_path):
    frames = []
    with Image.open(webp_path) as im:
        try:
            while True:
                frame = im.convert('RGB')
                frames.append(np.array(frame)[:, :, ::-1])  # RGB to BGR
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    return frames

def compute_sim(f1, f2):
    assert f1.shape == f2.shape, f"Shape mismatch: {f1.shape} vs {f2.shape}"
    assert f1.shape[2] == 3, f"Expected 3 channels, got {f1.shape[2]}"
    assert f1.dtype == np.float32, f"Expected float32, got {f1.dtype}"
    assert f2.dtype == np.float32, f"Expected float32, got {f2.dtype}"
    # Flatten to 1D vectors
    v1 = f1.flatten()
    v2 = f2.flatten()
    eps = 1e-8
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
    return cos_sim

def detect_loops(frames, min_len=4, max_len=40, top_k=10):
    n = len(frames)
    candidates = []
    # Preprocess frames: grayscale float32 and downscale to 128x128 (float32)
    processed_frames = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    # Build 3-channel composite frames: R=prev, G=curr, B=next (looping)
    n = len(processed_frames)
    composite_frames = []
    motion_energies = []
    for idx in range(n):
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        r = processed_frames[prev_idx]
        g = processed_frames[idx]
        b = processed_frames[next_idx]
        composite = np.stack([r, g, b], axis=-1)
        composite_frames.append(composite)
        if idx == n - 1:
            motion_energies.append(0.0)
        else:
            # waving hand 0.97
            # body shake 1.78
            # running 15.0
            # breathing 0.56
            # talking 0.39
            motion_energy = np.mean(np.abs(processed_frames[next_idx] - processed_frames[idx]))
            motion_energies.append(motion_energy)
    max_motion_energy = 20.0
    normalized_motion_energies = []
    for me in motion_energies:
        nme = min(me / max_motion_energy, 1.0)
        normalized_motion_energies.append(nme)
    for i in range(n):
        for j in range(i+min_len, min(i+max_len, n)):
            # Compare composite frames directly
            start_comp = composite_frames[i].astype(np.float32)
            end_comp = composite_frames[j].astype(np.float32)
            cos_sim = compute_sim(start_comp, end_comp)
            length_penalty = evaluate_length_penalty(j - i)
            motion_energy = motion_energies[i]
            nme = normalized_motion_energies[i]
            similarity_correction = (1.0 - cos_sim) * nme * 0.5
            score = cos_sim + similarity_correction - length_penalty  # Higher cosine similarity is better
            # print(f"Loop ({i},{j}): Cosine similarity={cos_sim:.4f} (t={t1-t0:.3f}s)")
            candidates.append((score, i, j, cos_sim, length_penalty, motion_energy))
    # Sort by score descending
    candidates.sort(reverse=True)
    return candidates[:top_k]

if __name__ == "__main__":

    # For batch processing, change the pattern below to 'sample-*.webp' or similar
    if not files:
        print('No files found.')
        import sys
        sys.exit(1)

    output_dir = Path('data/loops')
    output_dir.mkdir(parents=True, exist_ok=True)

    # import matplotlib.pyplot as plt
    for webp_path in files:
        print(f"Processing {webp_path}")
        frames = extract_frames(webp_path)
        print(f"Extracted {len(frames)} frames from {webp_path}")
        loops = detect_loops(frames)
        loop_json = []
        for score, i, j, cos_sim, len_penalty, motion_energy in loops:
            loop_json.append({
                "start": int(i),
                "end": int(j),
                "score": float(score),
                "cos_sim": float(cos_sim),
                "length": int(j - i),
                "length_penalty": float(len_penalty),
                "motion_energy": float(len_penalty)
            })
        json_name = f"{webp_path.stem}.loop.json"
        json_path = output_dir / json_name
        with open(json_path, "w") as f:
            json.dump(loop_json, f, indent=2)
        print(f"Saved loop candidates: {json_path}")
        for idx, (score, i, j, cos_sim, len_penalty, motion_energy) in enumerate(loops):
            print(f"Loop candidate: start={i}, end={j}, score={score:.6f}, COS_SIM={cos_sim:.6f}, LEN={int(j - i)}, LEN_PENALTY={len_penalty:.6f}, MOTION_ENERGY={motion_energy:.6f}")
            if idx != 0:
                continue  # For now, only save the top candidate
            # Extract loop frames (seamless looping: frames[i:j])
            loop_frames = frames[i:j]
            # Convert BGR (OpenCV) to RGB for PIL
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in loop_frames]
            # Save as animated webp
            out_name = f"{webp_path.stem}.loop.webp"
            out_path = output_dir / out_name
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=40,  # default duration, can be improved by extracting from original
                loop=0,
                lossless=True
            )
"""

