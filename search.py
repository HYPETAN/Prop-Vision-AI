import torch
import open_clip
import chromadb

# 1. Connect to our existing database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="real_estate_properties")

# 2. Load the CLIP Model and fine-tuned weights
print("Loading CLIP Model and fine-tuned weights for search...")
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

try:
    model.load_state_dict(torch.load("clip_realestate_finetuned.pt"))
    print("Successfully loaded fine-tuned weights into search engine!")
except FileNotFoundError:
    print("Warning: Fine-tuned weights not found. Using base pre-trained weights.")

model.eval()

def search_properties(query_text, top_k=3):
    print(f"\n--- Searching for: '{query_text}' ---")
    
    # Embed the text query using the fine-tuned model
    text_input = tokenizer([query_text])
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).tolist()[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[text_embedding],
        n_results=top_k
    )
    
    # Display Results
    for i in range(top_k):
        print(f"\nResult #{i+1} (ID: {results['ids'][0][i]})")
        print(f"Distance Score: {results['distances'][0][i]:.4f} (Lower is better)")
        print(f"Image Path: {results['metadatas'][0][i]['image_path']}")
        print(f"Caption: {results['metadatas'][0][i]['description']}")

# Test it!
search_properties("A bathroom with a bathtub")