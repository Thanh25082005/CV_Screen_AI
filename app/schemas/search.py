"""Search and matching request/response schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SearchType(str, Enum):
    """Type of search to perform."""

    HYBRID = "hybrid"  # BM25 + Vector with RRF
    KEYWORD = "keyword"  # BM25 only
    SEMANTIC = "semantic"  # Vector only


class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str = Field(..., description="Search query text (Semantic/Vector)")
    keyword_query: Optional[str] = Field(None, description="Specific keyword query for BM25 (Optional)")
    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Type of search to perform"
    )
    expand_query: bool = Field(
        default=True, description="Whether to expand query using LLM"
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results")
    min_experience_years: Optional[float] = Field(
        None, ge=0, description="Minimum years of experience"
    )
    required_skills: List[str] = Field(
        default_factory=list, description="Must-have skills"
    )
    preferred_skills: List[str] = Field(
        default_factory=list, description="Nice-to-have skills"
    )
    education_level: Optional[str] = Field(
        None, description="Minimum education level"
    )
    location: Optional[str] = None

    # Advanced filters
    include_sections: List[str] = Field(
        default_factory=list,
        description="Sections to search in (e.g., 'Experience', 'Projects')",
    )


class ChunkMatch(BaseModel):
    """Individual chunk match in search results."""

    chunk_id: str
    section: str
    subsection: Optional[str] = None
    content: str
    enriched_content: Optional[str] = None
    score: float = Field(..., description="Relevance score")
    match_type: str = Field(..., description="How this chunk matched (keyword/semantic)")


class SearchResult(BaseModel):
    """Individual search result with candidate info and matched chunks."""

    candidate_id: str
    full_name: str
    email: Optional[str] = None
    headline: Optional[str] = None
    total_experience_years: Optional[float] = None
    top_skills: List[str] = Field(default_factory=list)
    
    # Scoring
    combined_score: float = Field(..., description="RRF combined score")
    keyword_rank: Optional[int] = Field(None, description="Rank in keyword search")
    semantic_rank: Optional[int] = Field(None, description="Rank in semantic search")
    
    # Matched chunks
    matched_chunks: List[ChunkMatch] = Field(default_factory=list)
    
    # Match quality
    skills_matched: List[str] = Field(default_factory=list)
    skills_missing: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Complete search response."""

    query: str
    expanded_queries: List[str] = Field(
        default_factory=list, description="LLM-expanded query variations"
    )
    search_type: SearchType
    total_results: int
    results: List[SearchResult]
    
    # Metadata
    search_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MatchRequest(BaseModel):
    """Match candidates against a job description."""

    job_title: str = Field(..., description="Job title")
    job_description: str = Field(..., description="Full job description text")
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: Optional[float] = None
    max_experience_years: Optional[float] = None
    education_requirements: Optional[str] = None
    location: Optional[str] = None
    
    # Matching preferences
    top_k: int = Field(default=20, ge=1, le=100)
    skill_weight: float = Field(default=0.4, ge=0, le=1)
    experience_weight: float = Field(default=0.3, ge=0, le=1)
    semantic_weight: float = Field(default=0.3, ge=0, le=1)


class MatchScore(BaseModel):
    """Detailed match scoring breakdown."""

    overall_score: float = Field(..., ge=0, le=100, description="Overall match percentage")
    skill_score: float = Field(..., ge=0, le=100)
    experience_score: float = Field(..., ge=0, le=100)
    semantic_score: float = Field(..., ge=0, le=100)
    education_score: float = Field(default=0, ge=0, le=100)


class MatchResult(BaseModel):
    """Individual match result."""

    candidate_id: str
    full_name: str
    email: Optional[str] = None
    headline: Optional[str] = None
    total_experience_years: Optional[float] = None
    
    # Match details
    match_score: MatchScore
    skills_matched: List[str] = Field(default_factory=list)
    skills_missing: List[str] = Field(default_factory=list)
    experience_summary: Optional[str] = None
    
    # Explanation
    match_explanation: str = Field(
        ..., description="LLM-generated explanation of why this candidate matches"
    )


class MatchResponse(BaseModel):
    """Complete matching response."""

    job_title: str
    total_candidates_evaluated: int
    total_matches: int
    matches: List[MatchResult]
    
    # Metadata
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
