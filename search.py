import torch
import open_clip
import chromadb

# 1. Connect to our existing database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="real_estate_properties")

# 2. Load the CLIP Model (it will load instantly this time from cache)
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def search_properties(query_text, top_k=3):
    print(f"\n--- Searching for: '{query_text}' ---")
    
    # Embed the text query
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
        print(f"Distance Score: {results['distances'][0][i]:.4f}")
        print(f"Image Path: {results['metadatas'][0][i]['image_path']}")
        print(f"Caption: {results['metadatas'][0][i]['description']}")

# Try a few different natural language searches!
search_properties("A bathroom with a bathtub")
search_properties("An empty bedroom with white walls")