# PDF RAG Chatbot with Streamlit and Hugging Face Embeddings

This project implements a Retrieval Augmented Generation (RAG) chatbot using Streamlit, allowing users to upload a PDF document and then ask questions about its content. The chatbot uses Hugging Face's `sentence-transformers` for generating L2-normalized text embeddings and FAISS for efficient cosine similarity search to retrieve relevant passages.

## Features
- Upload PDF documents.
- Automatically processes PDF text into chunks and builds a searchable FAISS index.
- Ask questions and retrieve the most relevant passages from the uploaded PDF.
- Utilizes L2 normalization for embeddings and `faiss.IndexFlatIP` for accurate cosine similarity search.
- Built with Streamlit for an interactive web interface.

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Local Installation and Running
1.  **Create project files:**
    Create a directory for your project.

2.  **Create `streamlit_app.py`:**
    Save the Streamlit application code (the Python code you provided) as `streamlit_app.py` in your project directory.

3.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the same directory with the following content:
    ```
    streamlit
    sentence-transformers
    faiss-cpu
    PyPDF2
torch
    ```

4.  **Install dependencies:**
    Open your terminal or command prompt, navigate to your project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```
    Your browser will automatically open to the Streamlit app (usually at `http://localhost:8501`).

### Usage
1.  **Upload a PDF:** Use the file uploader widget to select a PDF document.
2.  **Wait for Processing:** The app will process the PDF, chunk the text, and build a FAISS index. A success message will appear once completed.
3.  **Ask a Question:** Enter your question related to the PDF content in the text input field.
4.  **View Relevant Passages:** The app will display the top relevant passages from the PDF, along with their cosine similarity scores.

## Hugging Face API Token

For this specific project using `all-MiniLM-L6-v2` from `sentence-transformers` (which is a public model), **you do not need a Hugging Face API token (`HF_TOKEN`)**.

The model is publicly available and can be downloaded and used without authentication. If you were to use a private Hugging Face model or perform actions requiring authentication (like pushing models to the Hugging Face Hub), you would need to set the `HF_TOKEN` environment variable.
