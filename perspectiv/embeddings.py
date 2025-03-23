"""
Embeddings wrapper for the Perspectiv knowledge base.

This module provides a wrapper around SentenceTransformer to make it compatible
with LangChain's embeddings interface.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class PerspectivEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer to make it compatible with LangChain."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings wrapper.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings, one for each text
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to generate embedding for

        Returns:
            Query embedding
        """
        embedding = self.model.encode(text)
        return embedding.tolist() 