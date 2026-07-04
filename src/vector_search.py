"""
Vector Search using FAISS with Sentence Transformers
Optimized for low latency with pre-computed embeddings
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FastVectorSearch:
    
    def __init__(self):
        """Initialize vector search with pre-loaded documents and embeddings"""
        self.encoder = None
        self.index = None
        self.documents = []
        self.metadata = []
        self._is_ready = False
        
        # Performance tracking
        self.total_searches = 0
        self.total_latency_ms = 0.0
        
        # Load everything
        self._load_encoder()
        self._load_data()
        
        logger.info(f"Vector search initialized: documents={len(self.documents)}, "
                   f"ready={self._is_ready}")
    
    def _load_encoder(self):
        """Load sentence transformer encoder"""
        try:
            start_time = time.time()
            
            self.encoder = SentenceTransformer(
                config.EMBEDDING_MODEL,
                device="cpu"  # CPU for low latency
            )
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Encoder loaded in {load_time:.0f}ms: {config.EMBEDDING_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            raise
    
    def _load_data(self):
        """Load documents and build FAISS index"""
        try:
            # Load documents from data directory
            data_path = Path(config.DATA_DIR) / "sample_documents.json"
            
            if not data_path.exists():
                logger.warning(f"Documents file not found: {data_path}")
                self._load_empty_index()
                return
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract documents and metadata
            self.documents = []
            self.metadata = []
            
            for item in data:
                self.documents.append(item.get("text", ""))
                self.metadata.append({
                    "id": item.get("id", len(self.documents)),
                    "category": item.get("category", "general"),
                    "source": item.get("source", "unknown")
                })
            
            if not self.documents:
                logger.warning("No documents loaded")
                self._load_empty_index()
                return
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(self.documents)} documents...")
            start_time = time.time()
            
            embeddings = self.encoder.encode(
                self.documents,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )
            
            embed_time = (time.time() - start_time) * 1000
            logger.info(f"Embeddings generated in {embed_time:.0f}ms")
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized)
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            self._is_ready = True
            logger.info(f"FAISS index built: dimension={dimension}, "
                       f"documents={self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self._load_empty_index()
    
    def _load_empty_index(self):
        """Load empty index for fallback"""
        self._is_ready = False
        # Create empty index with dummy dimension
        dimension = 384  # MiniLM dimension
        self.index = faiss.IndexFlatIP(dimension)
        logger.warning("Empty index loaded - searches will return no results")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            
        Returns:
            List of dicts with text, metadata, and score
        """
        if not self._is_ready or not query or not query.strip():
            return []
        
        try:
            start_time = time.time()
            
            # Use config value if top_k not specified
            if top_k is None:
                top_k = config.TOP_K_RESULTS
            
            # Ensure top_k doesn't exceed available documents
            top_k = min(top_k, self.index.ntotal if self.index else 0)
            
            if top_k == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append({
                        "text": self.documents[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                    })
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.total_searches += 1
            self.total_latency_ms += latency_ms
            
            # Log occasional debug
            if self.total_searches % 50 == 0:
                avg_latency = self.total_latency_ms / self.total_searches
                logger.debug(f"Vector search stats: avg_latency={avg_latency:.2f}ms, "
                           f"searches={self.total_searches}")
            
            logger.debug(f"Vector search: '{query[:30]}...' -> {len(results)} results ({latency_ms:.0f}ms)")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def search_texts(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Search and return only text results
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of text strings
        """
        results = self.search(query, top_k)
        return [r["text"] for r in results]
    
    def search_with_threshold(self, query: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search with similarity threshold filter
        
        Args:
            query: Search query text
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of results with score >= threshold
        """
        results = self.search(query, top_k=10)  # Get more results first
        return [r for r in results if r["score"] >= threshold]
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add new documents to the index
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        """
        if not documents:
            return
        
        try:
            # Extract texts
            texts = [d.get("text", "") for d in documents]
            metadatas = [d.get("metadata", {}) for d in documents]
            
            # Generate embeddings
            embeddings = self.encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Add to existing index or create new
            if self._is_ready and self.index:
                # Add to existing index
                self.index.add(embeddings.astype(np.float32))
                self.documents.extend(texts)
                self.metadata.extend(metadatas)
            else:
                # Create new index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings.astype(np.float32))
                self.documents = texts
                self.metadata = metadatas
                self._is_ready = True
            
            logger.info(f"Added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if self.total_searches == 0:
            return {
                "total_searches": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "is_ready": self._is_ready,
                "document_count": len(self.documents)
            }
        
        return {
            "total_searches": self.total_searches,
            "avg_latency_ms": self.total_latency_ms / self.total_searches,
            "total_latency_ms": self.total_latency_ms,
            "is_ready": self._is_ready,
            "document_count": len(self.documents)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_searches = 0
        self.total_latency_ms = 0.0
    
    def get_document_count(self) -> int:
        """Get number of documents in index"""
        return len(self.documents)
    
    def is_ready(self) -> bool:
        """Check if vector search is ready"""
        return self._is_ready


# Singleton instance
_vector_search_instance = None


def get_vector_search() -> FastVectorSearch:
    """Get or create global vector search instance"""
    global _vector_search_instance
    
    if _vector_search_instance is None:
        _vector_search_instance = FastVectorSearch()
    
    return _vector_search_instance


# Example usage
if __name__ == "__main__":
    # Test vector search
    vs = FastVectorSearch()
    
    if vs.is_ready():
        # Test search
        results = vs.search("What is the weather like?", top_k=3)
        
        print("Vector Search Results:")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
            print()
        
        print("Performance Stats:")
        print(vs.get_performance_stats())
    else:
        print("Vector search not ready - no documents loaded")