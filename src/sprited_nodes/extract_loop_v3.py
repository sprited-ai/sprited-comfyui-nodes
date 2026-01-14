# file: custom_nodes/extract_loop_v3.py
#
# LoopExtractorNodeV3 – Find best loop in image batch
# ---------------------------------------------------
#   Takes an IMAGE batch and analyzes frame similarity to find
#   the best seamless loop candidate. Returns the starting frame
#   index and duration instead of the actual frames.
#
# Outputs:
#   • start_frame: Index of the first frame in the loop
#   • duration: Number of frames in the loop

import cv2
import numpy as np


class LoopExtractorNodeV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_len": ("INT", {"default": 4, "min": 1, "max": 40}),
                "max_len": ("INT", {"default": 40, "min": 4, "max": 120}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("start_frame", "duration")
    FUNCTION = "extract_loop"
    CATEGORY = "image/batch"
    DESCRIPTION = "Analyze image batch to find the best seamless loop candidate. Returns start frame index and duration."

    def extract_loop(self, image, min_len=4, max_len=40):
        """
        Analyze image batch and return the best loop parameters.
        
        Args:
            image: ComfyUI IMAGE tensor (B, H, W, C) in RGB format, values 0-1
            min_len: Minimum loop length in frames
            max_len: Maximum loop length in frames
            
        Returns:
            (start_frame, duration): Tuple of integers
        """
        
        # Helper: convert IMAGE batch to frames list
        def image_batch_to_frames(image_tensor):
            """Convert ComfyUI IMAGE tensor to list of BGR numpy arrays"""
            frames = []
            for i in range(image_tensor.shape[0]):
                frame_rgb = (image_tensor[i].cpu().numpy() * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
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
            
            # Preprocess frames: grayscale float32 and downscale to 128x128
            processed_frames = [
                cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), 
                          (128, 128), interpolation=cv2.INTER_AREA)
                for f in frames
            ]
            
            # Build 3-channel composite frames: R=prev, G=curr, B=next (looping)
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
                for j in range(i + min_len, min(i + max_len, n)):
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
        frames = image_batch_to_frames(image)
        
        if len(frames) < min_len:
            raise RuntimeError(f"Not enough frames ({len(frames)}) for loop extraction. Need at least {min_len}.")
        
        if len(frames) < max_len:
            # Adjust max_len if we don't have enough frames
            max_len = len(frames)
        
        best = detect_loops(frames, min_len, max_len, top_k=1)[0]
        score, start_frame, end_frame, cos_sim, length_penalty = best
        duration = end_frame - start_frame
        
        return (start_frame, duration)
