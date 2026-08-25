import torch
import open_clip
import chromadb
from datasets import load_dataset
import os

# 1. Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="real_estate_properties")

# 2. Load the CLIP Model structure
print("Loading CLIP Model and applying fine-tuned weights...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# --- PLUG IN YOUR FINE-TUNED WEIGHTS ---
model.load_state_dict(torch.load("clip_realestate_finetuned.pt"))
model.eval() # Set to evaluation mode for inference

# 3. Re-run indexing with the fine-tuned model
dataset = load_dataset("jashu66/realestate-image-dataset", split="train[:100]")

for i, item in enumerate(dataset):
    image = item['Image']           
    text_desc = item['Caption']     
    
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        # These embeddings are now generated using your custom fine-tuned weights!
        image_embedding = model.encode_image(image_input).tolist()[0]
    
    image_path = os.path.join("images", f"image_{i}.jpg")
    
    collection.upsert(  # Use upsert to overwrite existing IDs if re-running
        ids=[str(i)],
        embeddings=[image_embedding],
        metadatas=[{"description": text_desc, "image_path": image_path}]
    )

print("Database successfully re-indexed with fine-tuned model embeddings!")