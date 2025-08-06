"""
Factory module for creating RAG pipeline components.

This module provides factory classes that create and configure
the various components of the RAG system based on configuration.
"""

import os
from typing import Dict, Any

from .interfaces import DocumentProcessor, EmbeddingProvider, VectorStore, LanguageModel
from .document_processing import GeminiFlashProcessor
from .embeddings import NomicEmbeddingProvider, CohereEmbeddingProvider
from .vector_store import ChromaVectorStore, FAISSVectorStore
from .llm import GeminiLanguageModel
from .rag import RAGPipeline, PromptBuilder
from config import RAGConfig


class RAGFactory:
    """
    Factory class for creating and configuring RAG pipeline components.
    
    This factory provides methods to create properly configured instances
    of all RAG system components based on configuration settings.
    """
    
    @staticmethod
    def create_document_processor(config: RAGConfig) -> DocumentProcessor:
        """
        Create a document processor based on configuration.
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured document processor instance
            
        Raises:
            ValueError: If processor type is unsupported or API key is missing for Gemini
        """
        processor_type = config.document_processing.processor_type.lower()
        
        if processor_type == "gemini_flash":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required for Gemini Flash processor. "
                    "Set it with: export GOOGLE_API_KEY='your-api-key'"
                )
            
            return GeminiFlashProcessor(
                api_key=api_key,
                chunk_size=config.document_processing.chunk_size,
                chunk_overlap=config.document_processing.chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported document processor: {processor_type}. Supported: 'gemini_flash'")
    
    @staticmethod
    def create_embedding_provider(config: RAGConfig) -> EmbeddingProvider:
        """
        Create an embedding provider based on configuration.
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured embedding provider instance
            
        Raises:
            ValueError: If provider type is unsupported or API key is missing for Cohere
        """
        provider = config.embedding.provider.lower()
        
        if provider == "nomic":
            return NomicEmbeddingProvider(
                model_name=config.embedding.model_name,
                batch_size=config.embedding.batch_size
            )
        elif provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY environment variable is required for Cohere embedding provider. "
                    "Set it with: export COHERE_API_KEY='your-api-key'"
                )
            
            return CohereEmbeddingProvider(
                api_key=api_key,
                model_name=config.embedding.model_name,
                input_type=config.embedding.input_type,
                batch_size=config.embedding.batch_size
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Supported: 'nomic', 'cohere'")
    
    @staticmethod
    def create_vector_store(config: RAGConfig) -> VectorStore:
        """
        Create a vector store based on configuration.
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured vector store instance
        """
        # Get the vector store type from config, default to 'faiss' for better performance
        store_type = getattr(config.vector_store, 'type', 'faiss').lower()
        
        if store_type == 'chroma':
            return ChromaVectorStore(
                collection_name=config.vector_store.collection_name,
                persist_directory=config.vector_store.persist_directory
            )
        elif store_type == 'faiss':
            # Get dimension from config, default to 1024 (Cohere embed-english-v3.0)
            dimension = getattr(config.vector_store, 'dimension', 1024)
            return FAISSVectorStore(
                dimension=dimension,
                persist_directory=config.vector_store.persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}. Supported: 'chroma', 'faiss'")
    
    @staticmethod
    def create_language_model(config: RAGConfig) -> LanguageModel:
        """
        Create a language model based on configuration.
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured language model instance
            
        Raises:
            ValueError: If API key is not provided in environment
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for language model. "
                "Set it with: export GEMINI_API_KEY='your-api-key'"
            )
        
        return GeminiLanguageModel(
            api_key=api_key,
            model_name=config.language_model.model_name
        )
    
    @staticmethod
    def create_pipeline(config: RAGConfig) -> RAGPipeline:
        """
        Create a complete RAG pipeline with all components.
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured RAG pipeline instance
        """
        document_processor = RAGFactory.create_document_processor(config)
        embedding_provider = RAGFactory.create_embedding_provider(config)
        vector_store = RAGFactory.create_vector_store(config)
        language_model = RAGFactory.create_language_model(config)
        prompt_builder = PromptBuilder()
        
        return RAGPipeline(
            document_processor=document_processor,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            language_model=language_model,
            prompt_builder=prompt_builder
        )
    
    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> RAGPipeline:
        """
        Create a RAG pipeline from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configured RAG pipeline instance
        """
        config = RAGConfig.from_dict(config_dict)
        return RAGFactory.create_pipeline(config)
