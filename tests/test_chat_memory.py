"""
Tests for Chat ConversationMemory.

Tests the Redis-backed sliding window memory implementation.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import ChatMessage, MessageRole, CandidateCard
from app.services.chat.memory import ConversationMemory


class MockRedis:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
    
    async def rpush(self, key: str, value: str) -> int:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
        return len(self.data[key])
    
    async def lrange(self, key: str, start: int, end: int) -> list:
        if key not in self.data:
            return []
        if end == -1:
            return self.data[key][start:]
        return self.data[key][start:end + 1]
    
    async def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self.data:
            if end == -1:
                self.data[key] = self.data[key][start:]
            else:
                self.data[key] = self.data[key][start:end + 1]
    
    async def expire(self, key: str, seconds: int) -> None:
        pass  # Mock implementation
    
    async def delete(self, key: str) -> int:
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def llen(self, key: str) -> int:
        return len(self.data.get(key, []))
    
    async def ttl(self, key: str) -> int:
        return 3600 if key in self.data else -2
    
    async def close(self) -> None:
        pass


@pytest.fixture
def mock_redis():
    """Create a mock Redis instance."""
    return MockRedis()


@pytest.fixture
def memory(mock_redis):
    """Create a ConversationMemory with mock Redis."""
    mem = ConversationMemory(max_messages=5, ttl_seconds=3600)
    mem._redis = mock_redis
    return mem


class TestConversationMemory:
    """Tests for ConversationMemory class."""
    
    @pytest.mark.asyncio
    async def test_add_message_user(self, memory):
        """Test adding a user message."""
        session_id = "test-session-1"
        
        message = await memory.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content="Hello, find me a Python developer",
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, find me a Python developer"
        assert isinstance(message.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_add_message_with_candidates(self, memory):
        """Test adding an assistant message with candidate cards."""
        session_id = "test-session-2"
        
        candidates = [
            CandidateCard(
                candidate_id="cand-1",
                full_name="Nguyen Van A",
                headline="Python Developer",
                top_skills=["Python", "Django"],
            )
        ]
        
        message = await memory.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content="Found candidate Nguyen Van A",
            candidates=candidates,
        )
        
        assert message.role == MessageRole.ASSISTANT
        assert len(message.candidates) == 1
        assert message.candidates[0].full_name == "Nguyen Van A"
    
    @pytest.mark.asyncio
    async def test_get_history(self, memory):
        """Test retrieving conversation history."""
        session_id = "test-session-3"
        
        # Add multiple messages
        await memory.add_message(session_id, MessageRole.USER, "Message 1")
        await memory.add_message(session_id, MessageRole.ASSISTANT, "Response 1")
        await memory.add_message(session_id, MessageRole.USER, "Message 2")
        
        history = await memory.get_history(session_id)
        
        assert len(history) == 3
        assert history[0].content == "Message 1"
        assert history[1].content == "Response 1"
        assert history[2].content == "Message 2"
    
    @pytest.mark.asyncio
    async def test_sliding_window(self, memory):
        """Test that sliding window keeps only last N messages."""
        session_id = "test-session-4"
        
        # Add more messages than the window size (5)
        for i in range(8):
            await memory.add_message(session_id, MessageRole.USER, f"Message {i}")
        
        history = await memory.get_history(session_id)
        
        # Should only have last 5 messages
        assert len(history) == 5
        assert history[0].content == "Message 3"
        assert history[-1].content == "Message 7"
    
    @pytest.mark.asyncio
    async def test_clear_session(self, memory):
        """Test clearing a session."""
        session_id = "test-session-5"
        
        await memory.add_message(session_id, MessageRole.USER, "Hello")
        await memory.add_message(session_id, MessageRole.ASSISTANT, "Hi there")
        
        deleted = await memory.clear_session(session_id)
        assert deleted is True
        
        history = await memory.get_history(session_id)
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, memory):
        """Test clearing a session that doesn't exist."""
        deleted = await memory.clear_session("nonexistent-session")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_format_history_for_prompt(self, memory):
        """Test formatting history as string for LLM prompt."""
        session_id = "test-session-6"
        
        await memory.add_message(session_id, MessageRole.USER, "Find Python devs")
        await memory.add_message(session_id, MessageRole.ASSISTANT, "Found 3 candidates")
        await memory.add_message(session_id, MessageRole.USER, "Tell me more about the first one")
        
        formatted = await memory.format_history_for_prompt(session_id)
        
        assert "User: Find Python devs" in formatted
        assert "Assistant: Found 3 candidates" in formatted
        assert "User: Tell me more about the first one" in formatted
    
    @pytest.mark.asyncio
    async def test_get_session_info(self, memory):
        """Test getting session metadata."""
        session_id = "test-session-7"
        
        await memory.add_message(session_id, MessageRole.USER, "Hello")
        await memory.add_message(session_id, MessageRole.ASSISTANT, "Hi")
        
        info = await memory.get_session_info(session_id)
        
        assert info["session_id"] == session_id
        assert info["message_count"] == 2
        assert info["exists"] is True
