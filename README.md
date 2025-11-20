# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that can read any PDF and answer questions based on its content. This project demonstrates how AI models can combine **document retrieval** with **answer generation** for accurate, context-aware responses.

---

## Features

- ✅ Upload and process any PDF file  
- ✅ Chunking of PDF content for efficient retrieval  
- ✅ Embedding with vector representations (using Python embeddings + FAISS)  
- ✅ Retrieval of relevant passages for question answering  
- ✅ Summarization and multi-fact answering  
- ✅ Handles multiple queries without hallucination  

---

## How it Works

1. **PDF Processing:** Reads PDF content and splits it into manageable chunks.  
2. **Embedding:** Each chunk is converted into a vector embedding. Embeddings are normalized for accurate similarity calculation.  
3. **Vector Storage:** Chunks and embeddings are stored in **FAISS** for fast retrieval.  
4. **Retrieval + Generation:** User asks a question → system retrieves top relevant chunks → language model generates answer based on retrieved content.  

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
