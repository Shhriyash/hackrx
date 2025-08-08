"""
Embedding Factory

This module provides a factory for creating embedding providers based on
manual selection.
"""

import os
from typing import Optional, Union

from .base import BaseEmbeddingProvider
from .cohere_embeddings import CohereEmbeddingProvider
from .google_embeddings import GoogleEmbeddingProvider


class EmbeddingFactory:
    """
    Factory class for creating embedding providers based on manual selection.
    """
    
    @staticmethod
    def create_provider(
        provider: str,
        cohere_api_key: str = None,
        google_api_key: str = None,
        **kwargs
    ) -> BaseEmbeddingProvider:
        """
        Create an embedding provider of the specified type.
        
        Args:
            provider: Provider type ("cohere" or "google")
            cohere_api_key: Cohere API key (if None, uses environment variable)
            google_api_key: Google API key (if None, uses environment variable)
            **kwargs: Additional arguments passed to the provider constructor
            
        Returns:
            Configured embedding provider instance
            
        Raises:
            ValueError: If provider type is invalid or required credentials are missing
        """
        provider = provider.lower()
        
        if provider == "cohere":
            return EmbeddingFactory._create_cohere_provider(cohere_api_key, **kwargs)
        elif provider == "google":
            return EmbeddingFactory._create_google_provider(google_api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Use 'cohere' or 'google'")
    
    @staticmethod
    def _create_cohere_provider(api_key: str = None, **kwargs) -> CohereEmbeddingProvider:
        """
        Create a Cohere embedding provider.
        
        Args:
            api_key: Cohere API key (if None, uses environment variable)
            **kwargs: Additional arguments for the provider
            
        Returns:
            Configured Cohere embedding provider
            
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        if api_key is None:
            api_key = os.getenv("COHERE_API_KEY")
        
        if not api_key:
            raise ValueError("Cohere API key not provided and COHERE_API_KEY environment variable not set")
        
        return CohereEmbeddingProvider(api_key=api_key, **kwargs)
    
    @staticmethod
    def _create_google_provider(api_key: str = None, **kwargs) -> GoogleEmbeddingProvider:
        """
        Create a Google embedding provider.
        
        Args:
            api_key: Google API key (if None, uses environment variable)
            **kwargs: Additional arguments for the provider
            
        Returns:
            Configured Google embedding provider
            
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        # Google provider can work without explicit API key if gcloud is configured
        # But we'll check for the environment variable as a fallback
        if not api_key:
            print("Warning: No Google API key provided, relying on default credentials")
        
        return GoogleEmbeddingProvider(api_key=api_key, **kwargs)


# Convenience function for quick provider creation
def get_embedding_provider(provider: str, **kwargs) -> BaseEmbeddingProvider:
    """
    Convenience function to create an embedding provider.
    
    Args:
        provider: Provider type ("cohere" or "google")
        **kwargs: Additional arguments passed to the provider constructor
        
    Returns:
        Configured embedding provider instance
    """
    return EmbeddingFactory.create_provider(provider, **kwargs)
