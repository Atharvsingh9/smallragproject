from fastapi import FastAPI, HTTPException, Query
from app.rag.engine import RAGEngine

app = FastAPI()
rag_engine = RAGEngine()

@app.get("/query")
def query(question: str = Query(..., description="User question")):
    try:
        answer=rag_engine.generate_answer(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


