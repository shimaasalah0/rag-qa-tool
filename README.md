# rag-qa-tool
AI-powered PDF Q&amp;A tool using RAG
# 📄 RAG-Powered PDF Q&A Tool

An AI-powered web app that lets you upload any PDF and ask questions about it using natural language. Built with RAG (Retrieval-Augmented Generation) architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)

## 🎯 What It Does

Upload a PDF report, research paper, or any document — then ask it questions like:
- *"What are the main findings?"*
- *"Summarize the document"*
- *"Which countries were mentioned?"*

The app reads the document and answers using only the content inside it.

## 🏗️ How It Works (RAG Pipeline)
```
PDF → Split into chunks → Convert to embeddings → Store in ChromaDB
                                                        ↓
Question → Convert to embedding → Find top 3 similar chunks → Send to AI → Answer
```

1. **Document Ingestion** — PDF is loaded and split into 500-character chunks
2. **Embedding** — Each chunk is converted into a vector (numbers representing meaning) using HuggingFace's `all-MiniLM-L6-v2` model
3. **Vector Storage** — Embeddings are stored in ChromaDB (vector database)
4. **Retrieval** — When a question is asked, the 3 most semantically similar chunks are retrieved
5. **Generation** — Retrieved chunks + question are sent to Groq's Llama 3.3 model to generate a precise answer

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyPDF |

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-tool.git
cd rag-qa-tool
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install streamlit langchain langchain-community langchain-groq
pip install langchain-text-splitters chromadb pypdf python-dotenv
pip install sentence-transformers langchain-huggingface
```

**4. Set up your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

💡 Key Concepts Demonstrated

- RAG (Retrieval-Augmented Generation)** — AI answers based only on document content, not general knowledge
- Vector Embeddings** — Text converted to numerical representations for semantic search
- Vector Database** — ChromaDB stores and searches embeddings efficiently
- LLM Integration** — Groq's Llama model used for fast, free text generation
- Full-Stack Development** — Complete frontend + backend in a single Python app
