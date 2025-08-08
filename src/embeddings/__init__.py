"""
Embeddings package initialization

This package provides embedding providers for different services
with a unified interface through the BaseEmbeddingProvider abstract class.
"""

from .base import BaseEmbeddingProvider
from .cohere_embeddings import CohereEmbeddingProvider
from .google_embeddings import GoogleEmbeddingProvider
from .factory import EmbeddingFactory, get_embedding_provider

__all__ = [
    'BaseEmbeddingProvider',
    'CohereEmbeddingProvider', 
    'GoogleEmbeddingProvider',
    'EmbeddingFactory',
    'get_embedding_provider'
]
