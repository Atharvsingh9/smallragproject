from langchain_community.vectorstores import FAISS

class VectorStore:


    def __init__(self,embeddings):
        self.embeddings=embeddings
        self.store=None

    def build(self,texts):

        self.store=FAISS.from_texts(
            texts=texts, 
            embedding=self.embeddings.model)
        
    def search(self, query:str, k:int=3):

        if self.store is None:
            raise ValueError("Vector Store Not Initialized. Please build the store first.") 
        docs=self.store.similarity_search(query=query, k=k)
        return[doc.page_content for doc in docs]