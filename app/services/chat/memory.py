"""
Conversation Memory using Redis.

Implements sliding window memory to keep the last N messages per session.
"""

import json
import logging
from typing import List, Optional
from datetime import datetime

import redis.asyncio as redis

from app.config import get_settings
from app.schemas.chat import ChatMessage, MessageRole, CandidateCard

settings = get_settings()
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Redis-backed conversation memory with sliding window.
    
    Stores chat history as a JSON list, keeping only the last N messages
    to avoid context window overflow and reduce token costs.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_messages: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize conversation memory.
        
        Args:
            redis_url: Redis connection URL
            max_messages: Maximum messages to keep (sliding window)
            ttl_seconds: Session expiry time in seconds
        """
        self.redis_url = redis_url or settings.redis_url
        self.max_messages = max_messages or settings.chat_history_max_messages
        self.ttl_seconds = ttl_seconds or settings.chat_memory_ttl_seconds
        self._redis: Optional[redis.Redis] = None
        
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"chat:history:{session_id}"
    
    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        candidates: Optional[List[CandidateCard]] = None,
    ) -> ChatMessage:
        """
        Add a message to the conversation history.
        
        Implements sliding window by keeping only the last N messages.
        
        Args:
            session_id: Unique session identifier
            role: Message role (user/assistant/system)
            content: Message content
            candidates: Optional candidate cards referenced
            
        Returns:
            The created ChatMessage
        """
        r = await self._get_redis()
        key = self._get_key(session_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            candidates=candidates or [],
        )
        
        # Serialize message to JSON
        message_json = message.model_dump_json()
        
        # Add to list (right push)
        await r.rpush(key, message_json)
        
        # Trim to keep only last N messages (sliding window)
        await r.ltrim(key, -self.max_messages, -1)
        
        # Set/refresh TTL
        await r.expire(key, self.ttl_seconds)
        
        logger.debug(f"Added message to session {session_id}: {role.value}")
        
        return message
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of ChatMessage objects, oldest first
        """
        r = await self._get_redis()
        key = self._get_key(session_id)
        
        # Get all messages
        messages_json = await r.lrange(key, 0, -1)
        
        messages = []
        for msg_json in messages_json:
            try:
                msg_data = json.loads(msg_json)
                messages.append(ChatMessage(**msg_data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
        
        # Apply limit if specified
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        return messages
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if session existed and was cleared
        """
        r = await self._get_redis()
        key = self._get_key(session_id)
        
        deleted = await r.delete(key)
        
        logger.info(f"Cleared session {session_id}: {'success' if deleted else 'not found'}")
        
        return bool(deleted)
    
    async def get_session_info(self, session_id: str) -> dict:
        """Get metadata about a session."""
        r = await self._get_redis()
        key = self._get_key(session_id)
        
        message_count = await r.llen(key)
        ttl = await r.ttl(key)
        
        return {
            "session_id": session_id,
            "message_count": message_count,
            "ttl_seconds": ttl if ttl > 0 else None,
            "exists": message_count > 0,
        }
    
    async def format_history_for_prompt(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> str:
        """
        Format conversation history as a string for LLM prompts.
        
        Args:
            session_id: Unique session identifier
            max_messages: Optional limit
            
        Returns:
            Formatted conversation history string
        """
        messages = await self.get_history(session_id, limit=max_messages)
        
        if not messages:
            return ""
        
        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == MessageRole.USER else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted_lines)
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Singleton instance
_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """Get or create the conversation memory singleton."""
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
