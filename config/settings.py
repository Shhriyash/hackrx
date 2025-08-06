"""
Configuration settings for the RAG system.

This module contains configuration classes and settings for all components
of the RAG pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing."""
    processor_type: str = "docling"  # "docling" or "gemini_flash"
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: str = "nomic"  # "nomic" or "cohere"
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    batch_size: int = 32
    input_type: str = "search_document"  # For Cohere: "search_document", "search_query", "classification", "clustering"


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    collection_name: str = "pdf_chunks"
    persist_directory: Optional[str] = None


@dataclass
class LanguageModelConfig:
    """Configuration for language model."""
    model_name: str = "gemini-1.5-flash"


@dataclass
class RAGConfig:
    """Main configuration for the RAG system."""
    document_processing: DocumentProcessingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    language_model: LanguageModelConfig
    default_n_results: int = 10
    
    @classmethod
    def default(cls) -> 'RAGConfig':
        """Create a default configuration."""
        return cls(
            document_processing=DocumentProcessingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            language_model=LanguageModelConfig(),
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RAGConfig':
        """Create configuration from dictionary."""
        return cls(
            document_processing=DocumentProcessingConfig(
                **config_dict.get('document_processing', {})
            ),
            embedding=EmbeddingConfig(
                **config_dict.get('embedding', {})
            ),
            vector_store=VectorStoreConfig(
                **config_dict.get('vector_store', {})
            ),
            language_model=LanguageModelConfig(
                **config_dict.get('language_model', {})
            ),
            default_n_results=config_dict.get('default_n_results', 10)
        )
