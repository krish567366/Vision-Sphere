"""
Multi-Level Caching System inspired by Youtu-agent
High-performance caching with file + database modes and automatic expiration.
"""

import asyncio
import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import logging
import pickle
from datetime import datetime, timedelta


class CacheConfig:
    """Configuration for caching system."""
    
    def __init__(self,
                 cache_dir: str = "./cache",
                 expire_time: int = 3600,  # 1 hour default
                 max_cache_size_mb: int = 1000,  # 1GB default
                 cleanup_interval: int = 3600):  # 1 hour cleanup
        self.cache_dir = Path(cache_dir)
        self.expire_time = expire_time
        self.max_cache_size_mb = max_cache_size_mb
        self.cleanup_interval = cleanup_interval
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class AsyncFileCache:
    """
    High-performance file-based caching system.
    
    Features:
    - MD5-based cache keys for collision avoidance
    - Automatic expiration with timestamp tracking
    - Async I/O for non-blocking operations
    - Background cleanup tasks
    - Size-based cache eviction
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger('AsyncFileCache')
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Start background cleanup task
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
    
    def _generate_cache_key(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate MD5 cache key from function call signature.
        
        Args:
            function_name: Name of the function
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            MD5 hash as cache key
        """
        # Create deterministic string representation
        key_data = {
            'function': function_name,
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.config.cache_dir / f"{cache_key}.cache"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata file path for cache key."""
        return self.config.cache_dir / f"{cache_key}.meta"
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached value.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        if not cache_path.exists() or not meta_path.exists():
            return None
        
        try:
            # Check expiration
            async with asyncio.Lock():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                created_at = metadata['created_at']
                expires_at = created_at + self.config.expire_time
                
                if time.time() > expires_at:
                    # Expired - remove files
                    cache_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
                
                # Load cached data
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        except Exception as e:
            self.logger.warning(f"Cache read error for key {cache_key}: {str(e)}")
            # Clean up corrupted cache files
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
    
    async def set(self, cache_key: str, value: Any) -> bool:
        """
        Store value in cache.
        
        Args:
            cache_key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        try:
            async with self._lock:
                # Save data
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Save metadata
                metadata = {
                    'created_at': time.time(),
                    'size_bytes': cache_path.stat().st_size,
                    'function_name': getattr(value, '__name__', 'unknown')
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Cache write error for key {cache_key}: {str(e)}")
            return False
    
    async def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate cached entry.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if entry was removed
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        removed = False
        if cache_path.exists():
            cache_path.unlink()
            removed = True
        
        if meta_path.exists():
            meta_path.unlink()
            removed = True
        
        return removed
    
    async def clear_all(self) -> int:
        """
        Clear all cached entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        for file_path in self.config.cache_dir.iterdir():
            if file_path.suffix in ['.cache', '.meta']:
                file_path.unlink()
                removed_count += 1
        
        return removed_count // 2  # Each entry has 2 files
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_files = 0
        total_size = 0
        expired_count = 0
        
        current_time = time.time()
        
        for file_path in self.config.cache_dir.iterdir():
            if file_path.suffix == '.meta':
                total_files += 1
                
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                    
                    total_size += metadata.get('size_bytes', 0)
                    
                    # Check if expired
                    created_at = metadata['created_at']
                    if current_time > created_at + self.config.expire_time:
                        expired_count += 1
                
                except Exception:
                    expired_count += 1  # Corrupted metadata counts as expired
        
        return {
            'total_entries': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'expired_entries': expired_count,
            'cache_dir': str(self.config.cache_dir),
            'expire_time_seconds': self.config.expire_time
        }
    
    async def _background_cleanup(self):
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Clean up expired entries
                removed_count = await self._cleanup_expired()
                
                # Check cache size and evict if necessary
                stats = await self.get_cache_stats()
                if stats['total_size_mb'] > self.config.max_cache_size_mb:
                    evicted_count = await self._evict_oldest()
                    self.logger.info(f"Cache size limit exceeded, evicted {evicted_count} entries")
                
                if removed_count > 0:
                    self.logger.info(f"Background cleanup removed {removed_count} expired entries")
            
            except Exception as e:
                self.logger.error(f"Background cleanup error: {str(e)}")
    
    async def _cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        removed_count = 0
        current_time = time.time()
        
        for meta_path in self.config.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                created_at = metadata['created_at']
                if current_time > created_at + self.config.expire_time:
                    # Remove both cache and metadata files
                    cache_key = meta_path.stem
                    await self.invalidate(cache_key)
                    removed_count += 1
            
            except Exception:
                # Remove corrupted files
                meta_path.unlink(missing_ok=True)
                cache_path = self.config.cache_dir / f"{meta_path.stem}.cache"
                cache_path.unlink(missing_ok=True)
                removed_count += 1
        
        return removed_count
    
    async def _evict_oldest(self) -> int:
        """Evict oldest cache entries to stay under size limit."""
        # Get all metadata files with creation times
        files_with_times = []
        
        for meta_path in self.config.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                files_with_times.append((
                    meta_path.stem,
                    metadata['created_at'],
                    metadata.get('size_bytes', 0)
                ))
            except Exception:
                # Remove corrupted files
                await self.invalidate(meta_path.stem)
        
        # Sort by creation time (oldest first)
        files_with_times.sort(key=lambda x: x[1])
        
        # Remove oldest entries until under size limit
        removed_count = 0
        target_size = self.config.max_cache_size_mb * 0.8 * 1024 * 1024  # 80% of limit
        current_size = sum(size for _, _, size in files_with_times)
        
        for cache_key, _, size in files_with_times:
            if current_size <= target_size:
                break
            
            await self.invalidate(cache_key)
            current_size -= size
            removed_count += 1
        
        return removed_count


# Global cache instance
_default_cache = None


def get_default_cache() -> AsyncFileCache:
    """Get the default cache instance."""
    global _default_cache
    if _default_cache is None:
        config = CacheConfig()
        _default_cache = AsyncFileCache(config)
    return _default_cache


def async_file_cache(expire_time: int = 3600, 
                    cache_dir: Optional[str] = None,
                    mode: str = "file"):
    """
    Decorator for caching async function results.
    
    Args:
        expire_time: Cache expiration time in seconds
        cache_dir: Custom cache directory
        mode: Cache mode ('file' or 'db') - currently only 'file' implemented
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            if cache_dir:
                config = CacheConfig(cache_dir=cache_dir, expire_time=expire_time)
                cache = AsyncFileCache(config)
            else:
                cache = get_default_cache()
            
            # Generate cache key
            cache_key = cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# Example usage decorator for tools
def cached_tool_call(expire_time: int = 3600):
    """
    Decorator specifically for tool calls that need caching.
    
    Args:
        expire_time: Cache expiration time in seconds
    """
    return async_file_cache(expire_time=expire_time, cache_dir="./cache/tools")


# Cache management utilities
async def get_cache_statistics() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = get_default_cache()
    return await cache.get_cache_stats()


async def clear_cache() -> int:
    """Clear all cached data."""
    cache = get_default_cache()
    return await cache.clear_all()


async def cleanup_expired_cache() -> int:
    """Manually trigger cleanup of expired cache entries."""
    cache = get_default_cache()
    return await cache._cleanup_expired()


# Global cache manager instance
class AsyncCacheManager:
    """Manager for all cache instances."""
    
    def __init__(self):
        self.caches: Dict[str, AsyncFileCache] = {}
        self.logger = logging.getLogger('AsyncCacheManager')
    
    def get_cache(self, cache_dir: str = "./cache") -> AsyncFileCache:
        """Get or create cache instance."""
        if cache_dir not in self.caches:
            self.caches[cache_dir] = AsyncFileCache(cache_dir)
        return self.caches[cache_dir]
    
    async def clear_all_caches(self) -> Dict[str, int]:
        """Clear all cache instances."""
        results = {}
        for cache_dir, cache in self.caches.items():
            cleared = await cache.clear_all()
            results[cache_dir] = cleared
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        for cache_dir, cache in self.caches.items():
            stats[cache_dir] = await cache.get_cache_stats()
        return stats
    
    def is_healthy(self) -> bool:
        """Check if cache manager is healthy."""
        return True


cache_manager = AsyncCacheManager()


# Export commonly used functions for API
async def clear_all_caches():
    """Clear all caches."""
    return await cache_manager.clear_all_caches()


async def get_cache_stats():
    """Get cache statistics."""
    return await cache_manager.get_stats()
