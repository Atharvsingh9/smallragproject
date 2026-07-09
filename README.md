# RAG FastAPI App

A small Retrieval-Augmented Generation API that loads a PDF, chunks the text, stores embeddings in FAISS, and answers questions with Groq through LangChain.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_ACCESS_TOKEN=optional_huggingface_token_here
```

## Run

```powershell
uvicorn app.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/query?question=What is this document about?
```

## Docker

This project can also be run with Docker.

```powershell
docker build -f Docker/dockerfile -t rag-fastapi .
docker run --env-file .env -p 8000:8000 rag-fastapi
```
