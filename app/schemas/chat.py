"""Chat API request/response schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a chat message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class CandidateCard(BaseModel):
    """Mini-profile card for candidate citations."""
    
    candidate_id: str
    full_name: str
    headline: Optional[str] = None
    email: Optional[str] = None
    total_experience_years: Optional[float] = None
    top_skills: List[str] = Field(default_factory=list)
    match_score: Optional[float] = None


class RetrievedChunk(BaseModel):
    """Debug info: A chunk of CV text retrieved from the database."""
    
    chunk_id: str
    candidate_name: str
    section: str = Field(..., description="CV section (Experience, Education, Skills, etc.)")
    content: str = Field(..., description="Original text from CV")
    score: float = Field(..., description="Relevance score (0-1)")
    match_type: str = Field(default="hybrid", description="How matched: keyword, semantic, hybrid")



class ChatMessage(BaseModel):
    """Individual chat message."""
    
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    candidates: List[CandidateCard] = Field(
        default_factory=list,
        description="Candidate cards referenced in this message"
    )


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    
    session_id: str
    message: ChatMessage
    sources: List[CandidateCard] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Response containing chat history."""
    
    session_id: str
    messages: List[ChatMessage]
    total_messages: int


class TransformedQuery(BaseModel):
    """Result of query transformation."""
    
    search_query: str = Field(..., description="Optimized query for vector search (Legacy, mapped from semantic_query)")
    semantic_query: str = Field(..., description="Expanded descriptive sentence for vector search")
    keyword_string: str = Field(..., description="Optimized keywords for BM25 search")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured filters (location, experience, skills)"
    )
    is_search_needed: bool = Field(
        default=True,
        description="Whether a candidate search is needed"
    )
    intent: str = Field(
        default="search",
        description="User intent: search, summarize, compare, or chat"
    )
    explanation: Optional[str] = Field(None, description="Explanation of the transformation")
