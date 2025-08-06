"""
ChromaDB vector store implementation.

This module provides vector storage and retrieval capabilities using ChromaDB
for efficient similarity search in the RAG pipeline.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ..interfaces import VectorStore


class ChromaVectorStore(VectorStore):
    """
    Vector store implementation using ChromaDB.
    
    This class provides methods to store document embeddings and perform
    similarity searches using ChromaDB as the backend storage system.
    """
    
    def __init__(self, collection_name: str = "pdf_chunks", 
                 persist_directory: Optional[str] = None):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (optional)
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
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
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def exists(self) -> bool:
        """
        Check if the vector store contains any documents.
        
        Returns:
            True if the collection contains documents, False otherwise
        """
        return len(self.collection.get()["documents"]) > 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        all_data = self.collection.get()
        return {
            "collection_name": self.collection_name,
            "document_count": len(all_data["documents"]),
            "has_embeddings": len(all_data.get("embeddings", [])) > 0
        }
    
    def delete_collection(self) -> None:
        """Delete the current collection and all its data."""
        self.client.delete_collection(name=self.collection_name)
        # Recreate the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query embedding.
        
        This method is an alias for query but returns results in a different format
        that's compatible with the RAG pipeline.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        results = self.query(query_embedding, top_k)
        
        # Transform results to the expected format
        search_results = []
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        for i in range(len(documents)):
            result = {
                "content": documents[i],
                "text": documents[i],  # Alternative key for backward compatibility
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0.0
            }
            search_results.append(result)
        
        return search_results

    def update_document(self, document_id: str, document: str, 
                       embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Update a specific document in the collection.
        
        Args:
            document_id: Unique identifier for the document
            document: Updated document text
            embedding: Updated embedding vector
            metadata: Updated metadata dictionary
        """
        self.collection.update(
            ids=[document_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata]
        )
