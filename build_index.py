import torch
import open_clip
import chromadb
from datasets import load_dataset
import os

# 1. Initialize the ChromaDB Client
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="real_estate_properties")

# 2. Load the CLIP Model (this might take a minute to download the first time)
print("Loading CLIP Model (ViT-B-32)...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 3. Load your dataset
print("Loading Dataset...")
dataset = load_dataset("jashu66/realestate-image-dataset", split="train[:100]")

# 4. Generate Embeddings and Store in ChromaDB
print("Generating Embeddings...")
for i, item in enumerate(dataset):
    image = item['Image']           
    text_desc = item['Caption']     
    
    # Preprocess and embed the image
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input).tolist()[0]
    
    # Store in ChromaDB 
    image_path = os.path.join("images", f"image_{i}.jpg")
    
    collection.add(
        ids=[str(i)],
        embeddings=[image_embedding],
        metadatas=[{"description": text_desc, "image_path": image_path}]
    )
    if (i + 1) % 10 == 0:
        print(f"Stored item {i+1}/100 in ChromaDB")
        
print("Database successfully built!")