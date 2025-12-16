import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import io
from transformers import pipeline

# -----------------------------
# Load Models (SAFE FOR STREAMLIT CLOUD)
# -----------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=100  # shorter answers
    )

    return embed_model, llm

embed_model, llm = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def read_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(io.BytesIO(file.read()))
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def build_faiss(chunks):
    """Build a FAISS index from text chunks."""
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def retrieve(query, index, chunks, k=3):
    """Retrieve top-k relevant chunks from FAISS index."""
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question):
    """Generate a simple, concise answer using the LLM."""
    prompt = f"""
You are a very helpful teacher.

Answer the question using ONLY the information below.
Explain in very simple English like I am 5 years old.
Use short sentences and simple words.

Information:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt, max_new_tokens=500)
    return response[0]["generated_text"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 Talk to Your PDF (RAG App)")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with st.spinner("Processing PDF..."):
        text = read_pdf(pdf)
        chunks = chunk_text(text)
        index, stored_chunks = build_faiss(chunks)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Generating answer..."):
            retrieved = retrieve(question, index, stored_chunks)
            context = "\n\n".join(retrieved)
            answer = generate_answer(context, question)

        st.subheader("🧠 Answer")
        st.write(answer)
