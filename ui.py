import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Prop-Vision-AI", layout="wide")
st.title("Prop-Vision-AI: Semantic Search")
st.markdown("Search for properties using natural language concepts instead of rigid filters.")

# 2. Connect to your FastAPI Backend
API_URL = "http://127.0.0.1:8000/search"

# 3. User Input
query = st.text_input("What kind of room are you looking for?", "A bright living room with large windows")
top_k = st.slider("Number of results to return", min_value=1, max_value=5, value=3)

# 4. Search Execution
if st.button("Search Properties"):
    with st.spinner("Querying the GenAI database..."):
        try:
            # Send the text to your FastAPI backend
            response = requests.post(API_URL, json={"query": query, "top_k": top_k})
            
            if response.status_code == 200:
                results = response.json()["results"]
                
                st.success(f"Found {len(results)} matches!")
                
                # Display the images side-by-side in columns
                cols = st.columns(len(results))
                
                for idx, res in enumerate(results):
                    with cols[idx]:
                        # Updated to remove the deprecated parameter
                        st.image(res["image_path"])
                        st.write(f"**Description:** {res['caption']}")
                        st.caption(f"Vector Distance: {res['distance']}")
            else:
                st.error("Error: Could not retrieve results from the backend.")
                
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Is your FastAPI server running in the other terminal?")