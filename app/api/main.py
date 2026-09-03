from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.eval.dataset import DatasetManager
from app.eval.runner import RAGEvaluator
from app.models.eval import EvalReport, EvalSample
from app.rag.engine import RAGEngine

app = FastAPI(
    title="Eval RAG API",
    description="Retrieval-Augmented Generation with Built-in Quality Evaluation",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag_engine = RAGEngine()

evaluator = RAGEvaluator(engine=rag_engine)


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: Optional[List[str]] = None
    latency_seconds: Optional[float] = None


class EvalRequest(BaseModel):
    samples: Optional[List[EvalSample]] = None
    limit: Optional[int] = None


@app.get("/query", response_model=QueryResponse)
def query(
    question: str = Query(..., description="User question to ask the RAG system"),
    include_contexts: bool = Query(
        False, description="Whether to include retrieved context chunks and latency in response"
    ),
):
    """Ask a question to the RAG system."""
    try:
        if include_contexts:
            result = rag_engine.query_with_context(question)
            return QueryResponse(
                question=question,
                answer=result["answer"],
                contexts=result["contexts"],
                latency_seconds=result["latency_seconds"],
            )
        else:
            answer = rag_engine.generate_answer(question)
            return QueryResponse(question=question, answer=answer)
    except Exception as e:
        print("❌ API ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/samples", response_model=List[EvalSample])
def get_eval_samples():
    """Retrieve the current benchmark evaluation dataset."""
    try:
        return DatasetManager.get_default_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval/run", response_model=EvalReport)
def run_evaluation(request: Optional[EvalRequest] = None):
    """Run an evaluation batch against the RAG pipeline."""
    try:
        samples = request.samples if (request and request.samples) else DatasetManager.get_default_dataset()
        if request and request.limit and request.limit > 0:
            samples = samples[: request.limit]
        report = evaluator.evaluate_dataset(samples=samples, verbose=False)
        return report
    except Exception as e:
        print("❌ EVAL ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

# Mount static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=RedirectResponse)
def read_root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/documents", response_model=List[str])
def list_documents():
    """List filenames in the data directory."""
    import os
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    try:
        return [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file to the data directory and trigger re-indexing."""
    import shutil, os
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    destination = os.path.join(data_dir, file.filename)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    rag_engine._initialize()
    return {"filename": file.filename, "status": "uploaded"}

@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Delete a document and re-index."""
    import os
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(file_path)
    rag_engine._initialize()
    return {"filename": filename, "status": "deleted"}

@app.post("/api/reindex")
def reindex():
    """Manually trigger re-indexing of all documents."""
    rag_engine._initialize()
    return {"status": "reindexed"}
