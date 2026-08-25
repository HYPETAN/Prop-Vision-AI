from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import open_clip
import chromadb

# 1. Initialize API
app = FastAPI(
    title="Prop-Vision-AI Search API",
    description="Multi-modal real estate search engine API",
    version="1.0.0"
)

# 2. Global variables to hold our model and DB (loaded on startup)
chroma_client = None
collection = None
model = None
tokenizer = None

# 3. Define the Request Data Structure using Pydantic
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

# 4. Load Models on Startup
@app.on_event("startup")
async def load_infrastructure():
    global chroma_client, collection, model, tokenizer
    print("Initializing Database & Models...")
    
    # Load DB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="real_estate_properties")
    
    # Load CLIP
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("Infrastructure loaded and ready!")
    
@app.get("/")
async def root():
    return {"message": "Welcome to the Prop-Vision-AI Engine. Go to /docs to test the API."}

# 5. Create the Search Endpoint
@app.post("/search")
async def search_properties(request: SearchRequest):
    try:
        # Embed the text query
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