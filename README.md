# Prop-Vision-AI: Multi-Modal Real Estate Search Engine

## Project Overview
Prop-Vision-AI is a proof-of-concept Multi-Modal Generative AI search engine built for the real estate domain. It allows users to query property listings using natural language and semantic concepts rather than traditional hard-coded filters (e.g., searching for "A modern bathroom with marble countertops" instead of clicking "Bathrooms: 1"). 

This project demonstrates an end-to-end Applied Machine Learning pipeline: data ingestion, vision-language embedding extraction, high-speed vector retrieval, and offline statistical evaluation.

## Tech Stack
*   **Deep Learning / Multi-Modal:** OpenAI CLIP (`ViT-B-32`), PyTorch, Hugging Face
*   **Vector Database:** ChromaDB
*   **Evaluation & Statistics:** Scikit-learn, SciPy, NumPy
*   **Data Processing:** Pandas, Pillow (PIL)

## Pipeline Architecture

1.  **Data Ingestion (`data_prep.py`):** 
    Pulls a sample dataset (`jashu66/realestate-image-dataset`) from Hugging Face containing interior/exterior property images and corresponding natural language captions.
2.  **Vectorization & Indexing (`build_index.py`):** 
    Passes images through the pre-trained CLIP Vision Transformer. Extracts dense vector embeddings and stores them in a local ChromaDB instance for low-latency retrieval.
3.  **Semantic Search (`search.py`):** 
    Encodes user text queries into the same shared vector space as the images, returning the Nearest Neighbor properties based on cosine distance.
4.  **Offline A/B Test Simulation (`ab_test.py`):** 
    Evaluates the theoretical business impact of the model. Compares the Mean Reciprocal Rank (MRR) of the Multi-Modal Search against a simulated baseline Keyword Search, utilizing a Paired T-Test to prove statistical significance.

## Demo & Output

### 1. Semantic Search Output
*Example of the model successfully understanding context and retrieving relevant images despite a limited 100-image sample pool.*

![Search Output](output/search_output.png) 

### 2. Statistical Evaluation (A/B Testing)
*Offline evaluation proving the statistical validity of the new model prior to deployment.*

![A/B Test Output](output/ab_test_output.png)

## Future Improvements & Scalability
While the current architecture serves as a functional MVP on a limited subset of 100 images, the system is designed to scale. Future iterations will include:
*   **Dataset Scaling:** Ingesting 50,000+ property images into ChromaDB to improve Nearest Neighbor accuracy and lower distance scores.
*   **API Deployment:** Wrapping the search engine in a FastAPI backend for frontend consumption.
*   **Fine-Tuning:** Fine-tuning the CLIP model on real estate-specific vocabulary (e.g., architectural styles like "Mid-Century Modern" or "Craftsman") to improve domain-specific retrieval precision.