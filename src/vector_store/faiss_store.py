"""
FAISS vector store implementation.

This module provides vector storage and retrieval capabilities using FAISS
for efficient similarity search in the RAG pipeline. FAISS is significantly
faster.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional

from ..interfaces import VectorStore


class FAISSVectorStore(VectorStore):
    """
    Vector store implementation using FAISS.
    
    This class provides methods to store document embeddings and perform
    ultra-fast similarity searches using FAISS as the backend storage system.
    FAISS is optimized for similarity search and clustering of dense vectors.
    """
    
    def __init__(self, dimension: int = 1024, persist_directory: Optional[str] = None):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of the embedding vectors (default 1024 for Cohere)
            persist_directory: Directory to persist the index (optional)
        """
        self.dimension = dimension
        self.persist_directory = persist_directory
        
        # Initialize FAISS index using IndexFlatIP (Inner Product for cosine similarity)
        # IndexFlatIP is fast and works well for most use cases
        self.index = faiss.IndexFlatIP(dimension)
        
        # Storage for documents, metadata, and IDs
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # Load existing index if persist_directory is provided
        if persist_directory and os.path.exists(persist_directory):
            self._load_index()
    
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
        # Convert embeddings to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings_array)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents, metadata, and IDs
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Persist if directory is specified
        if self.persist_directory:
            self._save_index()
    
    def query(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        if self.index.ntotal == 0:
            return {"documents": [], "metadatas": [], "distances": []}
        
        # Convert query embedding to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Perform search
        n_results = min(n_results, self.index.ntotal)
        distances, indices = self.index.search(query_array, n_results)
        
        # Extract results
        result_documents = []
        result_metadatas = []
        result_distances = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                result_documents.append(self.documents[idx])
                result_metadatas.append(self.metadatas[idx])
                result_distances.append(float(distances[0][i]))
        
        return {
            "documents": result_documents,
            "metadatas": result_metadatas,
            "distances": result_distances
        }
    
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
    
    def exists(self) -> bool:
        """
        Check if the vector store contains any documents.
        
        Returns:
            True if the index contains documents, False otherwise
        """
        return self.index.ntotal > 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        return {
            "collection_name": "faiss_index",
            "document_count": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP"
        }
    
    def delete_collection(self) -> None:
        """Delete the current index and all its data."""
        self.index.reset()
        self.documents.clear()
        self.metadatas.clear()
        self.ids.clear()
        
        # Remove persisted files if they exist
        if self.persist_directory:
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            data_path = os.path.join(self.persist_directory, "faiss_data.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(data_path):
                os.remove(data_path)
    
    def update_document(self, document_id: str, document: str, 
                       embedding: List[float], metadata: Dict[str, Any]) -> None:
        """
        Update a specific document in the collection.
        
        Note: FAISS doesn't support in-place updates efficiently.
        This implementation rebuilds the index which is expensive for large datasets.
        For production use, consider maintaining a separate mapping for updates.
        
        Args:
            document_id: Unique identifier for the document
            document: Updated document text
            embedding: Updated embedding vector
            metadata: Updated metadata dictionary
        """
        # Find the document index
        try:
            doc_index = self.ids.index(document_id)
        except ValueError:
            raise ValueError(f"Document with ID {document_id} not found")
        
        # Update the stored data
        self.documents[doc_index] = document
        self.metadatas[doc_index] = metadata
        
        # For FAISS, we need to rebuild the index (not efficient but simple)
        # In production, you might want to use a more sophisticated approach
        all_embeddings = []
        
        # Get all current embeddings (this is a limitation of this simple implementation)
        # In a real implementation, you'd store embeddings separately
        for i in range(len(self.documents)):
            if i == doc_index:
                all_embeddings.append(embedding)
            else:
                # This is a placeholder - in reality, you'd need to store embeddings
                # For now, we'll raise an error suggesting a full rebuild
                raise NotImplementedError(
                    "Document updates require storing original embeddings. "
                    "Consider rebuilding the entire index or implementing "
                    "a more sophisticated update mechanism."
                )
    
    def _save_index(self) -> None:
        """Save the FAISS index and associated data to disk."""
        if not self.persist_directory:
            return
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save documents, metadata, and IDs
        data_path = os.path.join(self.persist_directory, "faiss_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids,
                'dimension': self.dimension
            }, f)
    
    def _load_index(self) -> None:
        """Load the FAISS index and associated data from disk."""
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        data_path = os.path.join(self.persist_directory, "faiss_data.pkl")
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents, metadata, and IDs
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
                self.ids = data['ids']
                self.dimension = data.get('dimension', self.dimension)
