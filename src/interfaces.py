"""
Base interfaces for the RAG system components.

Interface Design Rationale: Abstract base classes provide contracts that ensure
all implementations follow the same API. This enables easy swapping of providers
(e.g., switching from Cohere to OpenAI embeddings) without changing dependent code.
The interface pattern also facilitates testing with mock implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DocumentProcessor(ABC):
    """Abstract base class for document processing components."""
    
    @abstractmethod
    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks with metadata.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            List of dictionaries containing chunk content and metadata
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class VectorStore(ABC):
    """Abstract base class for vector storage systems."""
    
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        pass
    
    @abstractmethod
    def query(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        pass
    
    @abstractmethod
    def exists(self) -> bool:
        """Check if the vector store contains any documents."""
        pass


class LanguageModel(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response based on the given prompt.
        
        Args:
            prompt: Input prompt for the language model
            
        Returns:
            Generated response text
        """
        pass
