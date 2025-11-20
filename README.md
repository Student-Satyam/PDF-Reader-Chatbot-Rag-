# PDF RAG Chatbot with Streamlit and Hugging Face Embeddings

This project implements a Retrieval Augmented Generation (RAG) chatbot using Streamlit, allowing users to upload a PDF document and then ask questions about its content. The chatbot uses Hugging Face's `sentence-transformers` for generating text embeddings and FAISS for efficient similarity search to retrieve relevant passages.

## Features
- Upload PDF documents.
- Automatically processes PDF text into chunks and builds a searchable FAISS index.
- Ask questions and retrieve the most relevant passages from the uploaded PDF.
- Built with Streamlit for an interactive web interface.

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Local Installation and Running
1.  **Clone the repository (or create project files):**
    If you're starting from scratch, create a directory for your project.

2.  **Create `streamlit_app.py`:**
    Save the Streamlit application code (provided in the previous output) as `streamlit_app.py` in your project directory.

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

### Deployment on Streamlit Cloud

1.  **Create a GitHub Repository:**
    - Create a new public GitHub repository.
    - Add your `streamlit_app.py` and `requirements.txt` files to this repository.

2.  **Deploy from Streamlit Cloud:**
    - Go to [Streamlit Cloud](https://share.streamlit.io/).
    - Click "New app" and connect to your GitHub repository.
    - Select the repository, the branch (e.g., `main`), and the `streamlit_app.py` file.
    - Click "Deploy!" Streamlit Cloud will handle installing the dependencies from `requirements.txt` and deploying your app.

## Hugging Face API Token (Important)

For this specific project using `all-MiniLM-L6-v2` from `sentence-transformers` (which is a public model), **you do not need a Hugging Face API token (`HF_TOKEN`)**.

The model is publicly available and can be downloaded and used without authentication.

If you were to use a private Hugging Face model or perform actions requiring authentication (like pushing models to the Hugging Face Hub), you would need to set the `HF_TOKEN` environment variable. For Streamlit Cloud, you would typically add this as a "Secret" in the app's advanced settings.
