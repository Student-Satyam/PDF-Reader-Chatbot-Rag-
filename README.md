📘 Talk to Your PDF (RAG App)

A Retrieval-Augmented Generation (RAG) app that lets you ask questions about your PDF documents and get answers in simple English. Built with Streamlit, FAISS, SentenceTransformers, and Hugging Face Transformers.

Features

Upload any PDF and extract its text.

Automatically split PDF text into smaller chunks for better retrieval.

Build a FAISS vector index for semantic search.

Ask questions in natural language and get AI-generated answers.

Answers are explained simply, like teaching a 5-year-old.

Demo Screenshot

(You can add a screenshot here showing the UI with PDF upload and question input.)

Installation

Clone the repo:

git clone https://github.com/yourusername/talk-to-pdf.git
cd talk-to-pdf


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install streamlit faiss-cpu sentence-transformers PyPDF2 transformers torch

Usage

Run the app:

streamlit run app.py


Upload a PDF.

Wait for it to process and build embeddings.

Enter a question in the input box.

Get a simple, AI-generated answer based on your PDF content.

How It Works

PDF Processing: Extracts text from your PDF using PyPDF2.

Text Chunking: Splits text into smaller chunks (default 400 words with 50-word overlap).

Embedding & Indexing: Uses SentenceTransformer (all-MiniLM-L6-v2) to generate embeddings and stores them in a FAISS index.

Retrieval: Finds the top k chunks most relevant to your question.

Answer Generation: Uses Flan-T5-base via Hugging Face transformers to generate a simple, coherent answer using only the retrieved context.

Code Highlights

Cached model loading to speed up Streamlit reloads:

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
    return embed_model, llm


Simple text chunking with overlap for better context retrieval.

FAISS vector search for semantic similarity.

Notes

Works best with text-based PDFs. Scanned PDFs may need OCR.

Keep PDFs under ~50 MB for smooth processing on Streamlit Cloud.

Model loading might take a few seconds, especially the first time.

Dependencies

Python 3.8+

Streamlit

FAISS

SentenceTransformers

PyPDF2

Transformers

Torch

License

MIT License – free to use and modify.
