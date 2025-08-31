"""
Advanced ML-Based Caching System
Semantic similarity caching with intelligent invalidation and relevance scoring.
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import math
from functools import wraps

# For semantic similarity (optional, graceful fallback)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SIMILARITY_AVAILABLE = True
except ImportError:
    SEMANTIC_SIMILARITY_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class CacheEntry:
    """Enhanced cache entry with semantic and usage metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    semantic_embedding: Optional[np.ndarray] = None
    relevance_score: float = 1.0
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def age_hours(self) -> float:
        """Age of cache entry in hours."""
        return (time.time() - self.created_at) / 3600
    
    @property
    def usage_frequency(self) -> float:
        """Usage frequency (accesses per hour)."""
        if self.age_hours == 0:
            return float(self.access_count)
        return self.access_count / self.age_hours
    
    def calculate_relevance(self, decay_factor: float = 0.1) -> float:
        """Calculate current relevance score based on age and usage."""
        # Exponential decay based on age
        age_decay = math.exp(-self.age_hours * decay_factor)
        
        # Usage boost
        usage_boost = min(math.log(1 + self.usage_frequency), 2.0)
        
        # Combined relevance
        return age_decay * (1 + usage_boost) * self.relevance_score


class SemanticCacheManager:
    """
    Advanced caching with semantic similarity matching and ML-based invalidation.
    """
    
    def __init__(self,
                 cache_dir: str = "./cache/semantic",
                 similarity_threshold: float = 0.85,
                 max_cache_size_gb: float = 5.0,
                 relevance_decay_factor: float = 0.05):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size_gb = max_cache_size_gb
        self.relevance_decay_factor = relevance_decay_factor
        
        # In-memory cache index
        self.cache_index: Dict[str, CacheEntry] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Semantic similarity model (lazy loaded)
        self._similarity_model: Optional[SentenceTransformer] = None
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "invalidations": 0,
            "total_size_bytes": 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the semantic cache manager."""
        await self._load_cache_index()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.logger.info("Semantic Cache Manager started")
    
    async def stop(self):
        """Stop the semantic cache manager."""
        self._shutdown_event.set()
        if self._cleanup_task:
            await self._cleanup_task
        await self._save_cache_index()
        self.logger.info("Semantic Cache Manager stopped")
    
    def _get_similarity_model(self) -> Optional[SentenceTransformer]:
        """Lazy load semantic similarity model."""
        if not SEMANTIC_SIMILARITY_AVAILABLE:
            return None
        
        if self._similarity_model is None:
            try:
                self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Semantic similarity model loaded")
            except Exception as e:
                self.logger.warning(f"Could not load similarity model: {e}")
                return None
        
        return self._similarity_model
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate semantic embedding for text."""
        model = self._get_similarity_model()
        if not model:
            return None
        
        try:
            embedding = model.encode([text])[0]
            return embedding.astype(np.float32)  # Reduce memory usage
        except Exception as e:
            self.logger.warning(f"Could not generate embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0
    
    def _generate_cache_key(self, data: Any, semantic_key: Optional[str] = None) -> str:
        """Generate cache key with optional semantic component."""
        # Base hash
        if isinstance(data, (str, bytes)):
            base_hash = hashlib.md5(str(data).encode()).hexdigest()
        else:
            base_hash = hashlib.md5(str(data).encode()).hexdigest()
        
        # Add semantic component if available
        if semantic_key:
            semantic_hash = hashlib.md5(semantic_key.encode()).hexdigest()[:8]
            return f"sem_{semantic_hash}_{base_hash}"
        
        return base_hash
    
    async def get(self, 
                  key: str, 
                  semantic_query: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> Optional[Any]:
        """
        Get cached value with semantic similarity matching.
        
        Args:
            key: Primary cache key
            semantic_query: Optional semantic query for similarity matching
            tags: Optional tags for filtering
            
        Returns:
            Cached value if found, None otherwise
        """
        # Direct key lookup
        if key in self.cache_index:
            entry = self.cache_index[key]
            
            # Tag filtering
            if tags and not any(tag in entry.tags for tag in tags):
                self.stats["misses"] += 1
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Load value
            try:
                value = await self._load_value(key)
                self.stats["hits"] += 1
                return value
            except Exception as e:
                self.logger.warning(f"Could not load cached value for {key}: {e}")
                await self._remove_entry(key)
                self.stats["misses"] += 1
                return None
        
        # Semantic similarity search
        if semantic_query and SEMANTIC_SIMILARITY_AVAILABLE:
            similar_entry = await self._find_similar_entry(semantic_query, tags)
            if similar_entry:
                try:
                    value = await self._load_value(similar_entry.key)
                    self.stats["semantic_hits"] += 1
                    self.logger.debug(f"Semantic cache hit for query: {semantic_query[:50]}...")
                    return value
                except Exception as e:
                    self.logger.warning(f"Could not load semantically similar value: {e}")
        
        self.stats["misses"] += 1
        return None
    
    async def set(self,
                  key: str,
                  value: Any,
                  semantic_key: Optional[str] = None,
                  ttl_hours: Optional[float] = None,
                  tags: Optional[List[str]] = None) -> bool:
        """
        Set cached value with semantic indexing.
        
        Args:
            key: Cache key
            value: Value to cache
            semantic_key: Optional semantic key for similarity matching
            ttl_hours: Time to live in hours
            tags: Optional tags for categorization
            
        Returns:
            True if successfully cached
        """
        try:
            # Generate semantic embedding
            embedding = None
            if semantic_key:
                embedding = self._generate_embedding(semantic_key)
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = len(str(value).encode())
            
            # Check cache size limits
            if not await self._check_size_limits(size_bytes):
                self.logger.warning("Cache size limit reached, could not add entry")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=None,  # Stored separately
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                semantic_embedding=embedding,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # Store value to disk
            await self._save_value(key, value)
            
            # Update index
            self.cache_index[key] = entry
            self.stats["total_size_bytes"] += size_bytes
            
            # Store embedding separately for fast similarity search
            if embedding is not None:
                self.embeddings_cache[key] = embedding
            
            self.logger.debug(f"Cached entry {key} (size: {size_bytes} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Could not cache entry {key}: {e}")
            return False
    
    async def _find_similar_entry(self, 
                                 query: str, 
                                 tags: Optional[List[str]] = None) -> Optional[CacheEntry]:
        """Find semantically similar cache entry."""
        if not self.embeddings_cache:
            return None
        
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return None
        
        best_similarity = 0.0
        best_entry = None
        
        for cache_key, embedding in self.embeddings_cache.items():
            entry = self.cache_index.get(cache_key)
            if not entry:
                continue
            
            # Tag filtering
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(query_embedding, embedding)
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                # Check relevance
                current_relevance = entry.calculate_relevance(self.relevance_decay_factor)
                if current_relevance > 0.1:  # Minimum relevance threshold
                    best_similarity = similarity
                    best_entry = entry
        
        return best_entry
    
    async def _check_size_limits(self, new_entry_size: int) -> bool:
        """Check if adding new entry would exceed size limits."""
        max_size_bytes = self.max_cache_size_gb * 1024 * 1024 * 1024
        
        if self.stats["total_size_bytes"] + new_entry_size > max_size_bytes:
            # Try to free space by removing low-relevance entries
            await self._cleanup_low_relevance_entries(new_entry_size)
            
            # Check again
            return self.stats["total_size_bytes"] + new_entry_size <= max_size_bytes
        
        return True
    
    async def _cleanup_low_relevance_entries(self, space_needed: int):
        """Remove entries with low relevance scores to free space."""
        # Calculate relevance for all entries
        entries_with_relevance = [
            (key, entry, entry.calculate_relevance(self.relevance_decay_factor))
            for key, entry in self.cache_index.items()
        ]
        
        # Sort by relevance (lowest first)
        entries_with_relevance.sort(key=lambda x: x[2])
        
        freed_space = 0
        for key, entry, relevance in entries_with_relevance:
            if freed_space >= space_needed:
                break
            
            if relevance < 0.1:  # Remove very low relevance entries
                await self._remove_entry(key)
                freed_space += entry.size_bytes
                self.stats["invalidations"] += 1
        
        self.logger.info(f"Freed {freed_space} bytes by removing low-relevance entries")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired and low-relevance entries."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                removed_count = 0
                freed_space = 0
                
                # Find entries to remove
                entries_to_remove = []
                for key, entry in list(self.cache_index.items()):
                    relevance = entry.calculate_relevance(self.relevance_decay_factor)
                    
                    # Remove if very old and unused
                    if (entry.age_hours > 168 and entry.access_count < 2) or relevance < 0.05:
                        entries_to_remove.append((key, entry))
                
                # Remove entries
                for key, entry in entries_to_remove:
                    await self._remove_entry(key)
                    removed_count += 1
                    freed_space += entry.size_bytes
                
                if removed_count > 0:
                    self.logger.info(
                        f"Periodic cleanup: removed {removed_count} entries, "
                        f"freed {freed_space / (1024*1024):.2f} MB"
                    )
                
                # Wait for next cleanup cycle
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _remove_entry(self, key: str):
        """Remove entry from cache and disk."""
        entry = self.cache_index.pop(key, None)
        if entry:
            self.stats["total_size_bytes"] -= entry.size_bytes
            
            # Remove from embeddings cache
            self.embeddings_cache.pop(key, None)
            
            # Remove file
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    async def _save_value(self, key: str, value: Any):
        """Save value to disk."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        def _write_file():
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_file)
    
    async def _load_value(self, key: str) -> Any:
        """Load value from disk."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        def _read_file():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_file)
    
    async def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.json"
        if not index_file.exists():
            return
        
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            for key, entry_data in index_data.items():
                # Reconstruct cache entry
                entry = CacheEntry(**entry_data)
                
                # Reconstruct numpy embedding
                if entry_data.get('semantic_embedding'):
                    embedding_data = entry_data['semantic_embedding']
                    entry.semantic_embedding = np.array(embedding_data, dtype=np.float32)
                    self.embeddings_cache[key] = entry.semantic_embedding
                
                self.cache_index[key] = entry
                self.stats["total_size_bytes"] += entry.size_bytes
            
            self.logger.info(f"Loaded {len(self.cache_index)} cache entries from disk")
            
        except Exception as e:
            self.logger.error(f"Could not load cache index: {e}")
    
    async def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "index.json"
        
        try:
            # Prepare serializable data
            index_data = {}
            for key, entry in self.cache_index.items():
                entry_dict = asdict(entry)
                
                # Convert numpy embedding to list
                if entry.semantic_embedding is not None:
                    entry_dict['semantic_embedding'] = entry.semantic_embedding.tolist()
                else:
                    entry_dict['semantic_embedding'] = None
                
                index_data[key] = entry_dict
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            self.logger.debug("Cache index saved to disk")
            
        except Exception as e:
            self.logger.error(f"Could not save cache index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        total_entries = len(self.cache_index)
        avg_relevance = 0.0
        
        if total_entries > 0:
            relevance_scores = [
                entry.calculate_relevance(self.relevance_decay_factor)
                for entry in self.cache_index.values()
            ]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        hit_rate = 0.0
        total_requests = self.stats["hits"] + self.stats["misses"]
        if total_requests > 0:
            hit_rate = self.stats["hits"] / total_requests
        
        semantic_hit_rate = 0.0
        if self.stats["hits"] > 0:
            semantic_hit_rate = self.stats["semantic_hits"] / self.stats["hits"]
        
        return {
            "total_entries": total_entries,
            "total_size_mb": self.stats["total_size_bytes"] / (1024 * 1024),
            "max_size_gb": self.max_cache_size_gb,
            "hit_rate": hit_rate,
            "semantic_hit_rate": semantic_hit_rate,
            "average_relevance": avg_relevance,
            "total_requests": total_requests,
            "semantic_similarity_available": SEMANTIC_SIMILARITY_AVAILABLE,
            **self.stats
        }
    
    async def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all entries matching any of the provided tags."""
        removed_count = 0
        
        for key, entry in list(self.cache_index.items()):
            if any(tag in entry.tags for tag in tags):
                await self._remove_entry(key)
                removed_count += 1
        
        self.logger.info(f"Invalidated {removed_count} entries by tags: {tags}")
        return removed_count
    
    async def find_similar_cached_results(self, 
                                        query: str, 
                                        tags: Optional[List[str]] = None,
                                        top_k: int = 5) -> List[Tuple[str, float, Any]]:
        """
        Find multiple similar cached results.
        
        Returns:
            List of (key, similarity_score, value) tuples
        """
        if not SEMANTIC_SIMILARITY_AVAILABLE or not self.embeddings_cache:
            return []
        
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        similarities = []
        
        for cache_key, embedding in self.embeddings_cache.items():
            entry = self.cache_index.get(cache_key)
            if not entry:
                continue
            
            # Tag filtering
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            similarity = self._calculate_similarity(query_embedding, embedding)
            if similarity > self.similarity_threshold:
                similarities.append((cache_key, similarity, entry))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Load values for top results
        results = []
        for cache_key, similarity, entry in similarities[:top_k]:
            try:
                value = await self._load_value(cache_key)
                results.append((cache_key, similarity, value))
            except Exception as e:
                self.logger.warning(f"Could not load similar result {cache_key}: {e}")
        
        return results


# Smart caching decorator with semantic similarity
def smart_semantic_cache(
    similarity_threshold: float = 0.85,
    ttl_hours: float = 24.0,
    tags: Optional[List[str]] = None,
    semantic_key_extractor: Optional[Callable[[Any], str]] = None
):
    """
    Decorator for semantic caching with intelligent invalidation.
    
    Args:
        similarity_threshold: Minimum similarity for cache hits
        ttl_hours: Time to live in hours
        tags: Cache entry tags
        semantic_key_extractor: Function to extract semantic key from args
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract semantic key
            semantic_key = None
            if semantic_key_extractor and args:
                try:
                    semantic_key = semantic_key_extractor(args[0])
                except Exception:
                    pass
            
            # Generate cache key
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = await semantic_cache_manager.get(
                cache_key, 
                semantic_query=semantic_key,
                tags=tags
            )
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await semantic_cache_manager.set(
                cache_key,
                result,
                semantic_key=semantic_key,
                ttl_hours=ttl_hours,
                tags=tags
            )
            
            return result
        
        return wrapper
    return decorator


# Global semantic cache manager
semantic_cache_manager = SemanticCacheManager()


async def initialize_semantic_cache():
    """Initialize the global semantic cache manager."""
    await semantic_cache_manager.start()
    logging.getLogger('SemanticCache').info("Semantic Cache Manager initialized")


async def shutdown_semantic_cache():
    """Shutdown the global semantic cache manager."""
    await semantic_cache_manager.stop()
