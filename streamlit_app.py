import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import io
from transformers import pipeline

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens=300,
        temperature=0.3
    )
    return embed_model, llm

embed_model, llm = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def read_pdf(file):
    text = ""
    reader = PdfReader(io.BytesIO(file.read()))
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def build_faiss(chunks):
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def retrieve(query, index, chunks, k=3):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question):
    prompt = f"""
You are a helpful teacher.

Use ONLY the information below.
Explain in very simple English like I am 5 years old.

Information:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt)
    return response[0]["generated_text"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 Talk to Your PDF (Full RAG App)")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    text = read_pdf(pdf)
    chunks = chunk_text(text)
    index, stored_chunks = build_faiss(chunks)
    st.success("PDF processed!")

    question = st.text_input("Ask a question:")

    if question:
        retrieved = retrieve(question, index, stored_chunks)
        context = "\n\n".join(retrieved)
        answer = generate_answer(context, question)

        st.subheader("🧠 Answer")
        st.write(answer)
