# get_embedding_function.py
from fastembed import TextEmbedding
from typing import List

class FastEmbedEmbeddings:
    def __init__(self):
        self.model = TextEmbedding()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return list(self.model.embed(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return list(self.model.embed([text]))[0]

def get_embedding_function():
    return FastEmbedEmbeddings()
