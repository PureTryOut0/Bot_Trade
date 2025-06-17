from functools import lru_cache
from datetime import datetime, timedelta
from typing import Callable, Any
from config import Config

class Cache:
    def __init__(self):
        self.cache = {}
        self.max_size = Config.CACHE_MAX_SIZE
        self.ttl = Config.CACHE_TTL.total_seconds()
        
    def get(self, key: str) -> Any:
        """Get value from cache with TTL check"""
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if datetime.now().timestamp() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            
        self.cache[key] = (value, datetime.now().timestamp())
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()

def cache_result(ttl: timedelta = None):
    """Decorator to cache function results"""
    if ttl is None:
        ttl = Config.CACHE_TTL
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache first
            result = Cache().get(cache_key)
            if result is not None:
                return result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            Cache().set(cache_key, result)
            return result
        return wrapper
    return decorator

def get_cache():
    """Get the cache instance"""
    return Cache()
