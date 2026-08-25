import os
from datasets import load_dataset

# 1. Create an 'images' directory if it doesn't exist
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the dataset
dataset = load_dataset("jashu66/realestate-image-dataset", split="train[:100]") 

print(dataset[0]) 

# 3. Save images locally inside the 'images' folder
for i, item in enumerate(dataset):
    # Construct the path: images/image_0.jpg
    save_path = os.path.join(output_dir, f"image_{i}.jpg")
    item['Image'].save(save_path)

print(f"\nSuccessfully saved {len(dataset)} images into the '{output_dir}/' folder!")