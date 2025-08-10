"""
Simple dictionary-based document cache for RAG pipeline.
Caches processed chunks and embeddings to avoid reprocessing identical documents.
"""

import hashlib
import time
import logging
import pickle
import os
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleDocumentCache:
    """
    Persistent dictionary-based cache for document processing results.
    Stores chunks, embeddings, and metadata to avoid reprocessing identical documents.
    Cache survives server restarts by persisting to disk.
    """
    
    def __init__(self, cache_file: str = "document_cache.pkl"):
        """Initialize the document cache with persistence."""
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_documents_cached": 0,
            "total_cache_size_mb": 0.0
        }
        
        # Load existing cache from disk
        self._load_cache()
        
        logger.info(f"Initialized SimpleDocumentCache with {len(self._cache)} cached documents")
        if len(self._cache) > 0:
            logger.info(f"Loaded cache with {self._stats['total_documents_cached']} documents ({self._stats['total_cache_size_mb']:.2f} MB)")
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self._cache = cache_data.get('cache', {})
                    self._stats = cache_data.get('stats', self._stats)
                logger.info(f"Cache loaded from {self.cache_file}")
            else:
                logger.info("No existing cache file found, starting with empty cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            logger.info("Starting with empty cache")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            cache_data = {
                'cache': self._cache,
                'stats': self._stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
    
    def _generate_document_key(self, document_url: str, content_preview: str = "") -> str:
        """
        Generate a unique cache key for a document based on filename.
        
        Why filename-based keys: Azure blob URLs contain SAS tokens and timestamps 
        that change frequently, making URL-based caching ineffective. By using 
        filename as the primary key, we can cache documents across different 
        URL variations (different SAS tokens, CDN URLs, etc.) for the same file.
        
        Args:
            document_url: The URL of the document
            content_preview: First few characters of content for validation (optional)
        
        Returns:
            Normalized cache key based on document name
        """
        # Extract filename from URL (handles both local files and URLs)
        if '/' in document_url:
            # For URLs like https://domain.com/path/filename.pdf?params
            url_path = document_url.split('?')[0]  # Remove query parameters
            filename = url_path.split('/')[-1]     # Get last part (filename)
        else:
            # For local files
            filename = document_url
        
        # URL decode the filename to handle encoded characters
        filename = urllib.parse.unquote(filename)
        
        # Use filename as the primary key (this makes cache work across different URLs)
        # Hash the filename to create a consistent, filesystem-safe cache key
        filename_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]
        
        # If content preview provided, create hybrid key for extra validation
        if content_preview:
            content_hash = hashlib.sha256(content_preview.encode()).hexdigest()[:8]
            return f"{filename_hash}_{content_hash}"
        
        logger.debug(f"Generated cache key '{filename_hash}' for document '{filename}'")
        return filename_hash
    
    def _estimate_cache_entry_size(self, chunks: List[Dict[str, Any]], 
                                 embeddings: List[List[float]]) -> float:
        """
        Estimate the memory size of a cache entry in MB.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embeddings
        
        Returns:
            Estimated size in MB
        """
        # Estimate chunk data size
        chunk_size = 0
        for chunk in chunks:
            chunk_size += len(str(chunk))
        
        # Estimate embeddings size (float64 = 8 bytes per dimension)
        embedding_size = len(embeddings) * len(embeddings[0]) * 8 if embeddings else 0
        
        total_bytes = chunk_size + embedding_size
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def is_cached(self, document_url: str, content_preview: str = "") -> bool:
        """
        Check if a document is already cached.
        
        Args:
            document_url: The URL of the document
            content_preview: First few characters of content for validation
        
        Returns:
            True if document is cached, False otherwise
        """
        cache_key = self._generate_document_key(document_url, content_preview)
        is_cached = cache_key in self._cache
        
        if is_cached:
            logger.info(f"Cache HIT for document: {document_url}")
            self._stats["cache_hits"] += 1
        else:
            logger.info(f"Cache MISS for document: {document_url}")
            self._stats["cache_misses"] += 1
        
        return is_cached
    
    def get_cached_data(self, document_url: str, content_preview: str = "") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached document data.
        
        Args:
            document_url: The URL of the document
            content_preview: First few characters of content for validation
        
        Returns:
            Cached data dictionary or None if not found
        """
        cache_key = self._generate_document_key(document_url, content_preview)
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            logger.info(f"Retrieved cached data for document: {document_url}")
            logger.info(f"Cached chunks: {len(cached_data['chunks'])}, "
                       f"Cached at: {cached_data['metadata']['cached_at']}")
            return cached_data
        
        return None
    
    def cache_document(self, document_url: str, chunks: List[Dict[str, Any]], 
                      embeddings: List[List[float]], chunk_lookup: Dict[int, Dict[str, Any]],
                      processing_time: float, content_preview: str = "") -> None:
        """
        Cache document processing results.
        
        Args:
            document_url: The URL of the document
            chunks: Processed document chunks
            embeddings: Generated embeddings
            chunk_lookup: Chunk lookup dictionary for neighbor retrieval
            processing_time: Time taken to process the document
            content_preview: First few characters of content for validation
        """
        cache_key = self._generate_document_key(document_url, content_preview)
        
        # Estimate cache entry size
        entry_size_mb = self._estimate_cache_entry_size(chunks, embeddings)
        
        # Create cache entry
        cache_entry = {
            "metadata": {
                "document_url": document_url,
                "cache_key": cache_key,
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "processing_time": processing_time,
                "cached_at": datetime.now().isoformat(),
                "entry_size_mb": entry_size_mb,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0
            },
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_lookup": chunk_lookup
        }
        
        # Store in cache
        self._cache[cache_key] = cache_entry
        
        # Update statistics
        self._stats["total_documents_cached"] += 1
        self._stats["total_cache_size_mb"] += entry_size_mb
        
        # Save cache to disk
        self._save_cache()
        
        logger.info(f"Cached document: {document_url}")
        logger.info(f"Cache entry size: {entry_size_mb:.2f} MB")
        logger.info(f"Total cached documents: {self._stats['total_documents_cached']}")
        logger.info(f"Total cache size: {self._stats['total_cache_size_mb']:.2f} MB")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        hit_rate = 0.0
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_requests > 0:
            hit_rate = (self._stats["cache_hits"] / total_requests) * 100
        
        stats = {
            **self._stats,
            "cache_hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "cached_document_keys": list(self._cache.keys())
        }
        
        return stats
    
    def list_cached_documents(self) -> List[Dict[str, Any]]:
        """
        List all cached documents with their metadata.
        
        Returns:
            List of cached document metadata
        """
        documents = []
        for cache_key, entry in self._cache.items():
            documents.append({
                "cache_key": cache_key,
                "document_url": entry["metadata"]["document_url"],
                "chunk_count": entry["metadata"]["chunk_count"],
                "cached_at": entry["metadata"]["cached_at"],
                "entry_size_mb": entry["metadata"]["entry_size_mb"],
                "processing_time": entry["metadata"]["processing_time"]
            })
        
        return documents
    
    def clear_cache(self) -> None:
        """Clear all cached documents."""
        cache_size = len(self._cache)
        self._cache.clear()
        
        # Reset stats but keep hit/miss counters for session tracking
        self._stats["total_documents_cached"] = 0
        self._stats["total_cache_size_mb"] = 0.0
        
        # Save cleared cache to disk
        self._save_cache()
        
        logger.info(f"Cleared cache: removed {cache_size} documents")
    
    def remove_document(self, document_url: str, content_preview: str = "") -> bool:
        """
        Remove a specific document from cache.
        
        Args:
            document_url: The URL of the document to remove
            content_preview: First few characters of content for validation
        
        Returns:
            True if document was removed, False if not found
        """
        cache_key = self._generate_document_key(document_url, content_preview)
        
        if cache_key in self._cache:
            entry_size = self._cache[cache_key]["metadata"]["entry_size_mb"]
            del self._cache[cache_key]
            
            # Update stats
            self._stats["total_documents_cached"] -= 1
            self._stats["total_cache_size_mb"] -= entry_size
            
            # Save updated cache to disk
            self._save_cache()
            
            logger.info(f"Removed document from cache: {document_url}")
            return True
        
        return False

# Global cache instance
_document_cache = SimpleDocumentCache()

def get_document_cache() -> SimpleDocumentCache:
    """Get the global document cache instance."""
    return _document_cache
