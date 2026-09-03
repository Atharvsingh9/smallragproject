import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    NVIDIA_MODEL_NAME,
    OCR_LANGUAGES,
    OCR_MODE,
    PDF_PATH,
    TOP_K,
)
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.loader import PDFLoader
from app.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """End-to-end Retrieval-Augmented Generation Engine with evaluation support."""

    def __init__(self):
        self.vector_store: VectorStore = None
        self.embeddings: EmbeddingModel = None
        self.llm = None
        self._initialize()

    def _initialize(self):
        load_dotenv(override=True)

        target_path = DATA_DIR if DATA_DIR.exists() else PDF_PATH
        text = PDFLoader(
            target_path,
            ocr_mode=OCR_MODE,
            ocr_languages=OCR_LANGUAGES,
        ).load()

        chunks = LangchainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)
        print(f"[RAGEngine] Text chunked into {len(chunks)} chunks.")

        self.embeddings = EmbeddingModel(EMBEDDING_MODEL_NAME)

        self.vector_store = VectorStore(self.embeddings)
        self.vector_store.build(chunks)

        api_key = os.getenv("NVIDIA_API_KEY")
        self.llm = ChatNVIDIA(model=NVIDIA_MODEL_NAME, api_key=api_key)
        print(f"[RAGEngine] Initialized LLM: {NVIDIA_MODEL_NAME}")

    def query_with_context(self, question: str, k: int = TOP_K) -> Dict[str, Any]:
        """Execute RAG query and return structured result including retrieved contexts and latency."""
        start_time = time.time()
        try:
            contexts = self.vector_store.search(question, k=k)
            combined_context = "\n\n".join(contexts)

            prompt_template = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained within the context, respond with "I don't know".

Context:
{combined_context}

Question:
{question}

Answer:"""

            response = self.llm.invoke(prompt_template)
            answer = response.content if hasattr(response, "content") else str(response)
            latency = time.time() - start_time

            return {
                "question": question,
                "answer": answer.strip(),
                "contexts": contexts,
                "combined_context": combined_context,
                "latency_seconds": round(latency, 3),
            }

        except Exception as e:
            logger.error(f"Error during RAG query: {repr(e)}")
            raise

    def generate_answer(self, question: str) -> str:
        """Generate answer for a question (backward-compatible)."""
        result = self.query_with_context(question)
        return result["answer"]