# Prop-Vision-AI: Multi-Modal Real Estate Search Engine

## Project Overview
Prop-Vision-AI is an end-to-end, production-grade Multi-Modal Generative AI search engine built for the real estate domain. It allows users to query property listings using natural language and semantic concepts rather than traditional hard-coded filters (e.g., searching for "A modern bathroom with marble countertops" instead of clicking "Bathrooms: 1"). 

This project demonstrates a decoupled microservice architecture: data ingestion, vision-language embedding extraction, high-speed vector retrieval, a production FastAPI backend, an interactive Streamlit frontend, and offline statistical evaluation.

## Tech Stack
* Deep Learning / Multi-Modal: OpenAI CLIP (ViT-B-32), PyTorch, Hugging Face
* Vector Database: ChromaDB
* Backend & API: FastAPI, Uvicorn, Pydantic
* Frontend UI: Streamlit, Requests
* Evaluation & Statistics: Scikit-learn, SciPy, NumPy
* Data Processing: Pandas, Pillow (PIL)

## System Architecture & Pipeline

1. Data Ingestion (data_prep.py): 
   Pulls a sample dataset (jashu66/realestate-image-dataset) from Hugging Face containing interior/exterior property images and corresponding natural language captions, organizing them locally.
2. Vectorization & Indexing (build_index.py): 
   Passes images through the pre-trained CLIP Vision Transformer, extracts dense vector embeddings, and stores them in a local ChromaDB instance for low-latency similarity search.
3. Backend REST API (app.py): 
   Asynchronously wraps the retrieval logic using FastAPI and Pydantic, exposing secure JSON endpoints (/search) to serve model inferences to downstream clients.
4. Frontend Microservice (ui.py): 
   A decoupled Streamlit user interface that consumes the FastAPI backend, allowing non-technical users to execute semantic searches and view side-by-side visual results.
5. Offline A/B Test Simulation (ab_test.py): 
   Evaluates theoretical business impact by comparing the Mean Reciprocal Rank (MRR) of the Multi-Modal Search against a baseline Keyword Search, utilizing a Paired T-Test to prove statistical significance.

## Getting Started & Local Installation

1. Clone the Repository & Install Dependencies:
git clone https://github.com/YOUR_USERNAME/Prop-Vision-AI.git
cd Prop-Vision-AI
pip install torch torchvision open_clip_torch chromadb fastapi uvicorn streamlit requests scikit-learn scipy pandas pillow datasets

2. Run the Ingestion & Indexing:
python data_prep.py
python build_index.py

3. Launch the Backend API:
uvicorn app:app --reload
(Interactive API docs available at http://127.0.0.1:8000/docs)

4. Launch the Frontend UI (in a separate terminal):
streamlit run ui.py
(Access the user interface at http://localhost:8501)

## Demo & Output

### 1. Full-Stack Search Interface (Streamlit + FastAPI)
![Streamlit UI](output/search_output.png)
*(Example of natural language query processed through the FastAPI backend and rendered on the UI)*

### 2. Statistical Evaluation (A/B Testing)
*Offline evaluation proving the statistical validity of the multi-modal model over keyword baselines.*
![A/B Test Output](output/ab_test_output.png)

## Future Improvements & Scalability
While the current architecture serves as a functional MVP on a limited local subset, the system is engineered to scale. Future iterations will include:
* Dataset Scaling: Ingesting 50,000+ property images into distributed ChromaDB clusters to lower distance scores and maximize recall.
* Model Fine-Tuning: Fine-tuning the CLIP model on real estate-specific vocabulary using PyTorch and Contrastive Loss to sharpen domain-specific retrieval precision.
* Containerization: Packaging the microservices using Docker and Docker Compose for seamless cloud orchestration on AWS EC2.