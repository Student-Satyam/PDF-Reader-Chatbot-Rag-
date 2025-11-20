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
    # 1. Do not embed empty strings or whitespace-only chunks, and add length check
    if not text or len(text.strip()) < 5:
        return np.array([])
    
    embeddings = embedding_model.encode([text], convert_to_numpy=True)
    
    # 3. Before normalization, check if the vector is all zeros or has norm < 1e-8.
    norm = np.linalg.norm(embeddings[0])
    
    # 4. If the vector has near-zero norm, return a zero vector (effectively skipping it for retrieval)
    if norm < 1e-8: # Use 1e-8 as requested
        return np.zeros_like(embeddings[0]) # Return a zero vector for non-meaningful inputs
        
    # 5. Use safe normalization: vector / (norm + 1e-8)
    normalized_embedding = embeddings[0] / (norm + 1e-8) 
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
    Splits a large text into smaller, overlapping chunks, with safety checks.
    1. Do not embed any empty strings or whitespace-only chunks.
    2. Add a check: if the chunk length is less than 5 characters, skip it.
    6. Ensure the chunking step removes duplicate chunks.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    seen_chunks = set() # To ensure unique chunks

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip() # Remove leading/trailing whitespace

        # Apply checks 1 and 2: skip empty/whitespace-only and short chunks
        if chunk and len(chunk) >= 5 and chunk not in seen_chunks: # Check 6: Remove duplicates
            chunks.append(chunk)
            seen_chunks.add(chunk)
        
        i += (chunk_size - chunk_overlap)
        # Prevent infinite loop if overlap is too large or chunk_size is too small
        if chunk_size - chunk_overlap <= 0 and i < len(words): 
            # If we haven't moved forward effectively, break to prevent infinite loop
            if i == (i - (chunk_size - chunk_overlap)) and i > 0: # Ensure we moved at least one word
                break

    return chunks

def build_faiss_index(chunks):
    """
    Generates L2-normalized embeddings for text chunks and builds a FAISS IndexFlatIP.
    Includes safety checks for near-zero norm embeddings.
    """
    if not chunks:
        return None, []

    # Generate embeddings for all chunks in a batch
    # The chunks list already comes pre-filtered by chunk_text for empty/short/duplicates
    chunk_embeddings_raw = embedding_model.encode(chunks, convert_to_numpy=True)

    final_embeddings_to_index = []
    final_chunks_for_index = []
    
    # Iterate through generated embeddings to apply safety checks
    for i, raw_embedding in enumerate(chunk_embeddings_raw):
        norm = np.linalg.norm(raw_embedding)
        
        # 3. Check if the vector has near-zero norm (using 1e-8 as requested)
        if norm < 1e-8:
            # 4. If near-zero norm, skip the chunk from indexing.
            # This prevents zero/corrupted vectors from entering the index.
            st.warning(f"Skipping chunk due to near-zero embedding norm: '{chunks[i][:50]}...' ")
            continue
        
        # 5. Use safe normalization: vector / (norm + 1e-8)
        normalized_embedding = raw_embedding / (norm + 1e-8)
        final_embeddings_to_index.append(normalized_embedding)
        final_chunks_for_index.append(chunks[i])

    if not final_embeddings_to_index:
        st.warning("No valid embeddings to build index from after filtering near-zero norm vectors.")
        return None, []

    final_embeddings_to_index = np.array(final_embeddings_to_index).astype('float32')
    embedding_dim = final_embeddings_to_index.shape[1]
    
    # Use IndexFlatIP for cosine similarity (since embeddings are L2 normalized)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(final_embeddings_to_index)

    return index, final_chunks_for_index

# --- Streamlit UI ---
st.title("PDF RAG Chatbot (Hugging Face Embeddings)")

st.markdown("Upload a PDF, then ask questions to retrieve relevant passages.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state or uploaded_file.name != st.session_state.get('uploaded_file_name'):
        st.session_state['uploaded_file_name'] = uploaded_file.name
        with st.spinner("Processing PDF and building index..."):
            raw_text = read_pdf(uploaded_file)
            if raw_text:
                st.session_state['chunks'] = chunk_text(raw_text) # Uses the improved chunk_text
                st.session_state['faiss_index'], st.session_state['indexed_chunks'] = build_faiss_index(st.session_state['chunks'])
                if st.session_state['faiss_index']:
                    st.success(f"PDF processed! Indexed {len(st.session_state['indexed_chunks'])} text chunks.")
                else:
                    st.error("Could not build FAISS index. No valid chunks were indexed.")
            else:
                st.error("Could not process the PDF. Please try another file.")
                st.session_state['faiss_index'] = None
                st.session_state['chunks'] = []
                st.session_state['indexed_chunks'] = []
    else:
        st.info("PDF already processed.")

    if st.session_state.get('faiss_index'):
        question = st.text_input("Ask a question about the PDF:")
        if question:
            with st.spinner("Searching for relevant information..."):
                question_embedding = get_embedding(question)
                
                # 7. Ensure retrieval does not return empty or zero-vector chunks
                # (This is implicitly handled because get_embedding returns np.array([]) or a zero vector
                # for problematic queries, and build_faiss_index skips problematic chunks.)
                if question_embedding.size > 0 and np.linalg.norm(question_embedding) > 1e-8: # Check for non-zero query embedding
                    D, I = st.session_state['faiss_index'].search(np.array([question_embedding]).astype('float32'), k=3) # Retrieve top 3
                    st.subheader("Relevant Passages:")
                    for rank, idx in enumerate(I[0]):
                        # D contains cosine similarity scores (inner product of L2 normalized vectors)
                        # Clip values to be within [-1, 1] for display in case of minor floating point inaccuracies
                        display_score = np.clip(D[0][rank], -1.0, 1.0)
                        st.write(f"**Passage {rank+1}:** (Cosine Similarity: {display_score:.4f})")
                        st.info(st.session_state['indexed_chunks'][idx]) # Use indexed_chunks here
                else:
                    st.warning("Could not generate a meaningful embedding for your question. Please try a more descriptive query.")
    else:
        st.warning("Please upload and process a PDF first.")

else:
    st.info("Awaiting PDF upload.")
