"""
RAG Pipeline implementation.

This module provides the main RAG pipeline that orchestrates document processing,
embedding generation, vector storage, retrieval, and response generation.
"""

from typing import List, Dict, Any, Optional
import json

from ..interfaces import DocumentProcessor, EmbeddingProvider, VectorStore, LanguageModel
from .prompt_builder import PromptBuilder


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components.
    
    This class provides a high-level interface for the complete RAG workflow,
    from document processing to answer generation.
    """
    
    def __init__(self, 
                 document_processor: DocumentProcessor,
                 embedding_provider: EmbeddingProvider,
                 vector_store: VectorStore,
                 language_model: LanguageModel,
                 prompt_builder: Optional[PromptBuilder] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            document_processor: Component for processing documents
            embedding_provider: Component for generating embeddings
            vector_store: Component for storing and retrieving vectors
            language_model: Component for generating responses
            prompt_builder: Optional custom prompt builder
        """
        self.document_processor = document_processor
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.language_model = language_model
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    def index_document(self, document_path: str) -> Dict[str, Any]:
        """
        Index a document into the vector store.
        
        Args:
            document_path: Path to the document to index
            
        Returns:
            Dictionary with indexing results and statistics
        """
        # Step 1: Process document into chunks
        print(f"Processing document: {document_path}")
        chunks = self.document_processor.process_document(document_path)
        
        if not chunks:
            return {"status": "error", "message": "No chunks extracted from document"}
        
        # Step 2: Extract text content and metadata
        texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [
            {"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"]}
            for chunk in chunks
        ]
        
        # Step 3: Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_provider.get_embeddings(texts)
        
        # Step 4: Store in vector database
        print("Storing chunks in vector database...")
        self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "document_path": document_path
        }
    
    def query(self, user_query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Query the RAG system and get an answer.
        
        Args:
            user_query: The user's question
            n_results: Number of chunks to retrieve for context
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Step 1: Generate query embedding
        query_embedding = self.embedding_provider.get_single_embedding(user_query)
        
        # Step 2: Retrieve relevant chunks
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        # Step 3: Format retrieved chunks
        retrieved_chunks = [
            {
                "chunk_id": meta.get("chunk_id", ""),
                "page_number": meta.get("page_number", "Unknown"),
                "content": doc
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        
        # Step 4: Build prompt
        prompt = self.prompt_builder.build_prompt(user_query, retrieved_chunks)
        
        # Step 5: Generate response
        response = self.language_model.generate_response(prompt)
        
        return {
            "query": user_query,
            "answer": response,
            "retrieved_chunks": len(retrieved_chunks),
            "context": retrieved_chunks
        }
    
    def is_indexed(self) -> bool:
        """
        Check if the vector store contains indexed documents.
        
        Returns:
            True if documents are indexed, False otherwise
        """
        return self.vector_store.exists()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline components.
        
        Returns:
            Dictionary containing component information
        """
        return {
            "document_processor": type(self.document_processor).__name__,
            "embedding_provider": type(self.embedding_provider).__name__,
            "vector_store": type(self.vector_store).__name__,
            "language_model": type(self.language_model).__name__,
            "is_indexed": self.is_indexed()
        }
    
    def clear_index(self) -> None:
        """Clear all indexed documents from the vector store."""
        if hasattr(self.vector_store, 'delete_collection'):
            self.vector_store.delete_collection()
    
    def batch_query(self, queries: List[str], n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            n_results: Number of chunks to retrieve for each query
            
        Returns:
            List of response dictionaries
        """
        return [self.query(query, n_results) for query in queries]
