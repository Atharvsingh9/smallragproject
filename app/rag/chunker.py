from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class LangchainTextChunker:


    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

