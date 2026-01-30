"""
Tests for Chat QueryTransformer.

Tests the LLM-powered query transformation.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.schemas.chat import TransformedQuery, ChatMessage, MessageRole
from app.services.chat.query_transformer import QueryTransformer


class TestQueryTransformer:
    """Tests for QueryTransformer class."""
    
    def test_transform_search_query(self):
        """Test transforming a search query."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            # Mock the Groq client response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''{
                "search_query": "Python Developer Backend",
                "filters": {"location": "Hanoi", "min_experience_years": 3},
                "is_search_needed": true,
                "intent": "search"
            }'''
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Tìm dev Python ở HN có 3 năm kinh nghiệm")
            
            assert result.search_query == "Python Developer Backend"
            assert result.filters.get("location") == "Hanoi"
            assert result.filters.get("min_experience_years") == 3
            assert result.is_search_needed is True
            assert result.intent == "search"
    
    def test_transform_chat_query(self):
        """Test transforming a simple chat message (no search needed)."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''{
                "search_query": "",
                "filters": {},
                "is_search_needed": false,
                "intent": "chat"
            }'''
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Xin chào")
            
            assert result.search_query == ""
            assert result.is_search_needed is False
            assert result.intent == "chat"
    
    def test_transform_summarize_intent(self):
        """Test transforming a summarize request."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''{
                "search_query": "Nguyễn Văn A",
                "filters": {},
                "is_search_needed": true,
                "intent": "summarize"
            }'''
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Tóm tắt hồ sơ của ứng viên Nguyễn Văn A")
            
            assert result.search_query == "Nguyễn Văn A"
            assert result.is_search_needed is True
            assert result.intent == "summarize"
    
    def test_transform_with_history(self):
        """Test query transformation with conversation history."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''{
                "search_query": "Java Developer Senior",
                "filters": {"min_experience_years": 5},
                "is_search_needed": true,
                "intent": "search"
            }'''
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            history = [
                ChatMessage(role=MessageRole.USER, content="Tìm dev Java"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Tôi tìm thấy 5 ứng viên"),
            ]
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Lọc những người có 5 năm trở lên", history)
            
            # Check that the API was called
            assert mock_client.chat.completions.create.called
            assert result.is_search_needed is True
    
    def test_transform_fallback_on_json_error(self):
        """Test fallback behavior when JSON parsing fails."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Invalid JSON response"
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Find Python developers")
            
            # Should fallback to using original message
            assert result.search_query == "Find Python developers"
            assert result.is_search_needed is True
            assert result.intent == "search"
    
    def test_transform_fallback_on_api_error(self):
        """Test fallback behavior when API call fails."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Find Java developers")
            
            # Should fallback to using original message
            assert result.search_query == "Find Java developers"
            assert result.is_search_needed is True
    
    def test_transform_handles_markdown_code_blocks(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        with patch('app.services.chat.query_transformer.Groq') as MockGroq:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''```json
{
    "search_query": "React Developer",
    "filters": {},
    "is_search_needed": true,
    "intent": "search"
}
```'''
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockGroq.return_value = mock_client
            
            transformer = QueryTransformer(api_key="test-key")
            result = transformer.transform("Find React developers")
            
            assert result.search_query == "React Developer"
            assert result.is_search_needed is True
