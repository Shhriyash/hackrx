"""
Cohere embedding provider implementation.

This module provides text embedding capabilities using Cohere's embedding API
for semantic similarity search and retrieval.
"""

from typing import List
import cohere

from ..interfaces import EmbeddingProvider


class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Cohere's text embedding API.
    
    This provider uses Cohere's embedding models to generate
    high-quality text embeddings for semantic search applications.
    """
    
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0", 
                 input_type: str = "search_document", batch_size: int = 96):
        """
        Initialize the Cohere embedding provider.
        
        Args:
            api_key: Cohere API key
            model_name: Name of the Cohere embedding model to use
            input_type: Type of input for embedding ("search_document", "search_query", "classification", "clustering")
            batch_size: Batch size for processing multiple texts (max 96 for Cohere)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.input_type = input_type
        self.batch_size = min(batch_size, 96)  # Cohere's max batch size
        
        # Initialize Cohere client
        self.client = cohere.Client(api_key)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as lists of floats
            
        Raises:
            Exception: If the Cohere API call fails
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Process a batch of texts and return their embeddings.
        
        Args:
            batch: List of text strings in the current batch
            
        Returns:
            List of embedding vectors for the batch
            
        Raises:
            Exception: If the Cohere API call fails
        """
        try:
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self.input_type
            )
            return response.embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings with Cohere: {str(e)}")
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.get_embeddings([text])[0]
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        This method temporarily switches to "search_query" input type
        for optimal query embedding generation.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector optimized for search queries
        """
        # Temporarily change input type for query
        original_input_type = self.input_type
        self.input_type = "search_query"
        
        try:
            embedding = self.get_single_embedding(query)
        finally:
            # Restore original input type
            self.input_type = original_input_type
        
        return embedding
    
    def get_model_info(self) -> dict:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "provider": "Cohere",
            "input_type": self.input_type,
            "batch_size": self.batch_size,
            "api_configured": bool(self.api_key)
        }
    
    def set_input_type(self, input_type: str) -> None:
        """
        Change the input type for future embeddings.
        
        Args:
            input_type: New input type ("search_document", "search_query", "classification", "clustering")
        """
        valid_types = ["search_document", "search_query", "classification", "clustering"]
        if input_type not in valid_types:
            raise ValueError(f"Invalid input_type. Must be one of: {valid_types}")
        
        self.input_type = input_type
