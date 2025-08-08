"""
Google GenAI Embedding Provider

This module provides an embedding provider that uses Google's GenAI
embedding models for generating text embeddings with parallel processing.
"""

import time
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai.types import EmbedContentConfig

from .base import BaseEmbeddingProvider

# Setup logging for this module
logger = logging.getLogger(__name__)


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Google's GenerativeAI embedding API.
    
    This provider uses Google's embedding models to generate
    high-quality text embeddings for semantic search applications.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-embedding-001"):
        """
        Initialize the Google embedding provider.
        
        Args:
            api_key: Google API key (if None, uses environment variable)
            model_name: Name of the Google embedding model to use
                       - gemini-embedding-001: Google's embedding model with 768 dimensions
        """
        self._model_name = model_name
        self.api_configured = True
        
        try:
            # Initialize the Google GenAI client
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                # Uses GOOGLE_API_KEY environment variable
                self.client = genai.Client()
            
            self.api_configured = True
        except Exception as e:
            print(f"Warning: Google API configuration issue: {e}")
            self.api_configured = False
            self.client = None
    
    def get_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Generate embeddings for a list of texts using parallel processing.
        
        Args:
            texts: List of text strings to embed
            task_type: Task type for embedding generation ("RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY")
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not self.api_configured or not self.client:
            raise Exception("Google API not properly configured")
        
        logger.info(f"Starting parallel embedding generation for {len(texts)} texts using {self._model_name} (task: {task_type})")
        
        # Use ThreadPoolExecutor for parallel processing without asyncio conflicts
        import concurrent.futures
        import threading
        
        # OPTIMIZATION: Increase workers for better throughput (within rate limits)
        max_workers = min(15, len(texts))  # Increased from 10 to 15 workers
        embeddings = [None] * len(texts)  # Pre-allocate list to maintain order
        
        def process_single_text(index_text_pair):
            index, text = index_text_pair
            max_retries = 3
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self._model_name,
                        contents=text,
                        config=EmbedContentConfig(
                            task_type=task_type,  # Use dynamic task type
                            output_dimensionality=768  # Reduced for better performance
                        )
                    )
                    return (index, response.embeddings[0].values)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Error generating embedding for text {index}: {str(e)}")
                    
                    # Simple retry delay
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        # Process in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            index_text_pairs = [(i, text) for i, text in enumerate(texts)]
            futures = {executor.submit(process_single_text, pair): pair[0] for pair in index_text_pairs}
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, embedding = future.result()
                    embeddings[index] = embedding
                except Exception as e:
                    original_index = futures[future]
                    logger.error(f"Failed to process text {original_index}: {e}")
                    raise
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings in parallel")
        return embeddings
    
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
        
        Uses task_type="RETRIEVAL_QUERY" for optimal query embedding generation.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector optimized for search queries
        """
        if not self.api_configured or not self.client:
            raise Exception("Google API not properly configured")
        
        try:
            response = self.client.models.embed_content(
                model=self._model_name,
                contents=query,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768  # Reduced for better performance
                )
            )
            return response.embeddings[0].values
        except Exception as e:
            raise Exception(f"Error generating query embedding with Google: {str(e)}")
    
    def get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple search queries using parallel processing.
        
        Uses task_type="RETRIEVAL_QUERY" for optimal query embedding generation.
        
        Args:
            queries: List of query texts to embed
            
        Returns:
            List of embedding vectors optimized for search queries
        """
        return self.get_embeddings(queries, task_type="RETRIEVAL_QUERY")
    
    def get_model_info(self) -> dict:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self._model_name,
            "provider": "Google",
            "api_configured": self.api_configured,
            "dimension": self.dimension
        }
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings generated by this provider."""
        return 768  # Reduced to 768 for optimal Google performance
    
    @property
    def model_name(self) -> str:
        """Return the model name used by this provider."""
        return self._model_name
    
    @property
    def provider_name(self) -> str:
        """Return the name of this embedding provider."""
        return "Google"
