from PIL import Image, ImageSequence
from transformers import CLIPProcessor, CLIPModel
import torch
import glob

def get_clip_scores(image, prompts, model, processor):
    # Process inputs
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, n_prompts]
        scores = logits_per_image.softmax(dim=1).squeeze().tolist()
    return list(zip(prompts, scores))

# Load animation.webp and process each frame
animation_path = "inputs/sample.webp"
prompts = ["greet", "idle", "run"]

model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

with Image.open(animation_path) as im:
    for idx, frame in enumerate(ImageSequence.Iterator(im)):
        frame_rgb = frame.convert("RGB")
        scores = get_clip_scores(frame_rgb, prompts, model, processor)
        print(f"Frame {idx}:")
        for prompt, score in scores:
            print(f"  Prompt: {prompt}, Score: {score:.4f}")
        print()
