from app.core.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, GROQ_MODEL_NAME, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.vectorstore import VectorStore
from app.rag.embeddings import EmbeddingModel

from langchain_groq import ChatGroq
from dotenv import load_dotenv

class RAGEngine:

    def __init__(self):
        self.vector_store=None
        self._initialize()

    def _initialize(self):
        load_dotenv(override=True)

        text=PDFLoader(PDF_PATH).load()

        Chunks=LangchainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)

        embeddings=EmbeddingModel(EMBEDDING_MODEL_NAME)

        self.vector_store=VectorStore(embeddings)
        self.vector_store.build(Chunks)

        self.llm=ChatGroq(model=GROQ_MODEL_NAME)

    def generate_answer(self, question:str):

        context=self.vector_store.search(question, k=TOP_K)
        combined_context="\n".join(context)


        prompt_template=f"""
        You are a helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, respond with "I don't know". 
        context: {combined_context}

        question: {question}
        answer:
        """

        result=self.llm.invoke(prompt_template)
        return result.content
