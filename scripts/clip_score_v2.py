
from PIL import Image, ImageSequence
from transformers import CLIPProcessor, CLIPModel
import torch
import glob
import matplotlib.pyplot as plt
from split_shot import detect_cuts_visual
import fire

def get_clip_scores_batch(images, prompts, model, processor):
    # Batch process all frames with all prompts
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [n_frames, n_prompts]
        scores = logits_per_image.softmax(dim=1).tolist()
    return scores  # shape: [n_frames, n_prompts]
def main(webp_path, prompts=None, batch_size=32, model_name="openai/clip-vit-base-patch16"):
    """
    Compute CLIP scores for all frames in a webp animation file.
    Args:
        webp_path: Path to the .webp animation file
        prompts: List of prompts (comma-separated string or list)
        batch_size: Batch size for CLIP scoring
        model_name: Huggingface model name for CLIP
    """
    if prompts is None:
        prompts = ["greet", "idle", "run"]
    elif isinstance(prompts, str):
        prompts = [p.strip() for p in prompts.split(",")]

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Collect all frames as RGB images
    frames_rgb = []
    with Image.open(webp_path) as im:
        for frame in ImageSequence.Iterator(im):
            frames_rgb.append(frame.convert("RGB"))

    num_frames = len(frames_rgb)
    all_scores = []
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch_frames = frames_rgb[start:end]
        scores_batch = get_clip_scores_batch(batch_frames, prompts, model, processor)
        # Normalize each frame's scores so they sum to 1
        for idx, scores in enumerate(scores_batch, start=start):
            total = sum(scores)
            if total > 0:
                norm_scores = [s / total for s in scores]
            else:
                norm_scores = scores
            all_scores.append(norm_scores)
            print(f"Frame {idx}:")
            for prompt, score in zip(prompts, norm_scores):
                print(f"  Prompt: {prompt}, Score: {score:.4f}")
            print()

    # Get diff scores using detect_cuts_visual
    diffs, _ = detect_cuts_visual(frames_rgb, k=3)
    diff_scores = [score for idx, score in diffs]
    # Pad diff_scores to match num_frames (diffs are between frames, so prepend a 0)
    if len(diff_scores) < num_frames:
        diff_scores = [0.0] + diff_scores

    # Plot the first derivative of probabilities for each prompt over timesteps and the diff scores
    all_scores_tensor = torch.tensor(all_scores)  # shape: [num_frames, num_prompts]
    timesteps = range(num_frames)
    plt.figure(figsize=(10, 6))
    derivatives = []
    for i, prompt in enumerate(prompts):
        # Compute first derivative (difference between consecutive frames)
        derivative = all_scores_tensor[:, i].numpy()
        derivative = derivative[1:] - derivative[:-1]
        abs_derivative = abs(derivative)
        derivatives.append(abs_derivative)
        plt.plot(timesteps[1:], derivative, label=f"{prompt} (derivative)")
    # Plot diff scores (scene cut diff)
    plt.plot(timesteps, diff_scores, label='Scene Cut Diff', color='black', linestyle='--', alpha=0.7)

    # Compute Euclidean norm of absolute derivatives for each frame (excluding first frame)
    if derivatives:
        # Stack derivatives: shape [num_prompts, num_frames-1]
        derivatives_arr = torch.stack([torch.tensor(d) for d in derivatives])  # shape: [num_prompts, num_frames-1]
        # Euclidean norm across prompts for each frame
        euclidean_norm = torch.sqrt(torch.sum(derivatives_arr ** 2, dim=0)).numpy()  # shape: [num_frames-1]
    else:
        euclidean_norm = None
    diff_scores_arr = torch.tensor(diff_scores[1:])  # skip first frame for alignment
    # Weighted average: 0.25 * euclidean_norm + 0.75 * diff_score
    if euclidean_norm is not None:
        weighted_avg = euclidean_norm + diff_scores_arr.numpy()
        plt.plot(timesteps[1:], weighted_avg, label='Weighted Avg (Euclidean Derivatives & Diff)', color='red', linewidth=2)

    plt.xlabel('Frame (Timestep)')
    plt.ylabel('First Derivative / Diff Score')
    plt.title('First Derivative of CLIP Prompt Probabilities and Scene Cut Diff Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/clip_scores_derivative_plot.png')
    print('Plot saved as clip_scores_derivative_plot.png')

if __name__ == "__main__":
    fire.Fire(main)
