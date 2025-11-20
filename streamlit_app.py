import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from PyPDF2 import PdfReader
import io

# --- Core PDF Chatbot Functions (from previous steps) ---

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def get_embedding(text):
    """Generates embeddings for a given text using the SentenceTransformer model, and applies L2 normalization."""
    if not text:
        return np.array([])
    embeddings = embedding_model.encode([text], convert_to_numpy=True)
    # Apply L2 normalization
    normalized_embedding = embeddings[0] / np.linalg.norm(embeddings[0])
    return normalized_embedding

def read_pdf(uploaded_file):
    """Reads text from an uploaded PDF file-like object."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text.strip()

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits a large text into smaller, overlapping chunks.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - chunk_overlap)
        if i >= len(words) and not chunks[-1] == " ".join(words[i - (chunk_size - chunk_overlap):]):
             break
    return chunks

def build_faiss_index(chunks):
    """
    Generates L2-normalized embeddings for text chunks and builds a FAISS IndexFlatIP.
    """
    if not chunks:
        return None, []

    # Generate embeddings for all chunks in a batch
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if not valid_chunks:
        st.warning("No valid chunks found to build index from.")
        return None, []

    # Encode and normalize embeddings
    chunk_embeddings_raw = embedding_model.encode(valid_chunks, convert_to_numpy=True)
    # Apply L2 normalization to all embeddings at once
    chunk_embeddings = chunk_embeddings_raw / np.linalg.norm(chunk_embeddings_raw, axis=1, keepdims=True)

    if chunk_embeddings.shape[0] == 0:
        return None, []

    embedding_dim = chunk_embeddings.shape[1]
    # Use IndexFlatIP for cosine similarity (since embeddings are L2 normalized)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings.astype('float32'))

    return index, valid_chunks

# --- Streamlit UI ---
st.title("PDF RAG Chatbot (Hugging Face Embeddings) satyam")

st.markdown("Upload a PDF, then ask questions to retrieve relevant passages.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state or uploaded_file.name != st.session_state.get('uploaded_file_name'):
        st.session_state['uploaded_file_name'] = uploaded_file.name
        with st.spinner("Processing PDF and building index..."):
            raw_text = read_pdf(uploaded_file)
            if raw_text:
                st.session_state['chunks'] = chunk_text(raw_text)
                st.session_state['faiss_index'], _ = build_faiss_index(st.session_state['chunks'])
                st.success(f"PDF processed! Found {len(st.session_state['chunks'])} text chunks.")
            else:
                st.error("Could not process the PDF. Please try another file.")
                st.session_state['faiss_index'] = None
                st.session_state['chunks'] = []
    else:
        st.info("PDF already processed.")

    if st.session_state.get('faiss_index'):
        question = st.text_input("Ask a question about the PDF:")
        if question:
            with st.spinner("Searching for relevant information..."):
                # Ensure question embedding is also L2 normalized
                question_embedding = get_embedding(question)
                if question_embedding.size > 0:
                    # Search with the normalized question embedding
                    D, I = st.session_state['faiss_index'].search(np.array([question_embedding]).astype('float32'), k=3) # Retrieve top 3
                    st.subheader("Relevant Passages:")
                    for rank, idx in enumerate(I[0]):
                        # D contains cosine similarity scores (inner product of L2 normalized vectors)
                        st.write(f"**Passage {rank+1}:** (Cosine Similarity: {D[0][rank]:.4f})")
                        st.info(st.session_state['chunks'][idx])
                else:
                    st.warning("Could not generate embedding for your question.")
    else:
        st.warning("Please upload and process a PDF first.")

else:
    st.info("Awaiting PDF upload.")
