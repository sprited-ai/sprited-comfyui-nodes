from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import glob
import matplotlib.pyplot as plt

def get_clip_scores(image_path, prompts, model_name="openai/clip-vit-base-patch16"):
    # Load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Process inputs
    inputs = processor(text=prompts, images=image, return_tensors="pt")

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

# Batch load images
images = [Image.open(f).convert("RGB") for f in image_files]

# Load model and processor once
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Prepare batch inputs
inputs = processor(text=prompts, images=images)

# Get batch embeddings and scores
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: [n_images, n_prompts]
    scores = logits_per_image.softmax(dim=1).cpu().numpy()

# Print and collect scores for plotting
all_scores = {prompt: [] for prompt in prompts}
for idx, image_file in enumerate(image_files):
    print(f"Processing {image_file}...")
    for j, prompt in enumerate(prompts):
        score = scores[idx, j]
        print(f"  Prompt: {prompt}, Score: {score:.4f}")
        all_scores[prompt].append(score)
    print()

# Plotting
time = list(range(len(image_files)))
for prompt in prompts:
    plt.plot(time, all_scores[prompt], label=prompt)
plt.xlabel('Time (image index)')
plt.ylabel('Probability')
plt.title('CLIP Probabilities over Time')
plt.legend()
plt.savefig('outputs/clip_probabilities.png')
plt.show()
