from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import open_clip
import chromadb

# 1. Initialize API
app = FastAPI(
    title="Prop-Vision-AI Search API",
    description="Multi-modal real estate search engine API backed by fine-tuned CLIP weights",
    version="2.0.0"
)

# 2. Global variables to hold our model and DB
chroma_client = None
collection = None
model = None
tokenizer = None

# 3. Define the Request Data Structure using Pydantic
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

# 4. Load Models and Fine-Tuned Weights on Startup
@app.on_event("startup")
async def load_infrastructure():
    global chroma_client, collection, model, tokenizer
    print("Initializing Database & Fine-Tuned Models...")
    
    # Load DB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="real_estate_properties")
    
    # Load CLIP Base Model Structure
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Load your custom fine-tuned weights!
    try:
        model.load_state_dict(torch.load("clip_realestate_finetuned.pt"))
        print("Successfully loaded fine-tuned weights into FastAPI server!")
    except FileNotFoundError:
        print("Warning: Fine-tuned weights not found. Running on base pre-trained weights.")
        
    model.eval()
    print("Infrastructure fully loaded and ready!")

# 5. Create the Search Endpoint
@app.post("/search")
async def search_properties(request: SearchRequest):
    try:
        # Embed the text query using the fine-tuned text encoder
        text_input = tokenizer([request.query])
        with torch.no_grad():
            text_embedding = model.encode_text(text_input).tolist()[0]
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[text_embedding],
            n_results=request.top_k
        )
        
        # Format the response for the frontend
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "rank": i + 1,
                "id": results['ids'][0][i],
                "distance": round(results['distances'][0][i], 4),
                "image_path": results['metadatas'][0][i]['image_path'],
                "caption": results['metadatas'][0][i]['description']
            })
            
        return {"query": request.query, "results": formatted_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))