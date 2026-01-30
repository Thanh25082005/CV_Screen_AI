import json
import logging
import hashlib
from typing import Any, Optional, Union
from datetime import timedelta

import redis.asyncio as redis
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class RedisCache:
    """Redis-based caching service."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[redis.Redis] = None
        
    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis
        
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a unique key based on data content."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
            
        hash_val = hashlib.md5(content.encode()).hexdigest()
        return f"cache:{prefix}:{hash_val}"
        
    async def get(self, prefix: str, key_data: Any) -> Optional[Any]:
        """Retrieve data from cache."""
        try:
            r = await self._get_redis()
            key = self._generate_key(prefix, key_data)
            data = await r.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
        
    async def set(
        self, 
        prefix: str, 
        key_data: Any, 
        value: Any, 
        expire_seconds: int = 300
    ) -> bool:
        """Store data in cache."""
        try:
            r = await self._get_redis()
            key = self._generate_key(prefix, key_data)
            await r.set(
                key, 
                json.dumps(value), 
                ex=expire_seconds
            )
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
        return False

# Singleton instance
_cache: Optional[RedisCache] = None

def get_cache() -> RedisCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
