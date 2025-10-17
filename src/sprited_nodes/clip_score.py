from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import glob

def get_clip_scores(image_path, prompts, model_name="openai/clip-vit-base-patch16"):
    # Load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process inputs
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, n_prompts]
        scores = logits_per_image.softmax(dim=1).squeeze().tolist()

    # Return scores for each prompt
    return list(zip(prompts, scores))

# Iterate over all sample_part*.webp files
image_files = glob.glob("outputs/sample_part*.webp")
prompts = ["greet", "idle", "run"]

for image_file in image_files:
    print(f"Processing {image_file}...")
    scores = get_clip_scores(image_file, prompts)
    for prompt, score in scores:
        print(f"  Prompt: {prompt}, Score: {score:.4f}")
    print()
