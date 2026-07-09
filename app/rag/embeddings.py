from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:

    def __init__(self,model_name:str):
        self.model=HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_documents(self,texts):
        return self.model.embed_documents(texts)
    
    def embed_query(self,query):
        return self.model.embed_query(query)