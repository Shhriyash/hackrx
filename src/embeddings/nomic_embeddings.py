"""
Nomic embedding provider implementation.

This module provides text embedding capabilities using the Nomic AI embedding model
for semantic similarity search and retrieval.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

from ..interfaces import EmbeddingProvider


class NomicEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Nomic AI's text embedding model.
    
    This provider uses the nomic-embed-text-v1.5 model to generate
    high-quality text embeddings for semantic search applications.
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", 
                 batch_size: int = 32):
        """
        Initialize the Nomic embedding provider.
        
        Args:
            model_name: Name of the Nomic embedding model to use
            batch_size: Batch size for processing multiple texts
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Process a batch of texts and return their embeddings.
        
        Args:
            batch: List of text strings in the current batch
            
        Returns:
            List of embedding vectors for the batch
        """
        # Tokenize the batch
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings.cpu().tolist()
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.get_embeddings([text])[0]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "embedding_dimension": self.model.config.hidden_size
        }
