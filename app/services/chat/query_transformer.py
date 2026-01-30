"""
Query Transformer using LLM.

Analyzes user messages to extract:
1. Search query - optimized for vector search
2. Structured filters - location, experience, skills
3. Intent - search, summarize, compare, or general chat
"""

import json
import logging
from typing import Optional, List

from groq import Groq

from app.config import get_settings
from app.schemas.chat import TransformedQuery, ChatMessage

settings = get_settings()
logger = logging.getLogger(__name__)


QUERY_TRANSFORM_PROMPT = """Role: You are an Expert Search Architect for a Recruitment RAG system.

Task: Transform the user's natural language query into a structured search object.

SCOPE CHECK (CRITICAL):
This system ONLY handles questions about:
✅ Candidates, CVs, resumes, job seekers
✅ Skills, experience, education, projects
✅ Recruitment, hiring, job matching

If the user asks about ANYTHING ELSE (politics, entertainment, general knowledge, weather, jokes, etc.), 
set "intent": "off_topic" and "is_search_needed": false.

Rules:
1. Keyword Expansion: Identify the core skills requested and provide at least 3 synonyms or related technologies.
2. Comparison Queries: If user asks to "compare" specifically named candidates (e.g. "Compare A and B"):
    - Set "intent": "compare"
    - Put BOTH names in "keyword_string" (e.g. "Nguyen Van A Vu Van B") to ensure both are retrieved.
    - Set "semantic_query" to "Compare skills and experience of Nguyen Van A and Vu Van B".
3. Metadata Extraction: Extract specific filters:
    - years_of_experience: (Number)
    - location: (City/Region)
    - job_title: (Target role)
4. Strictness Level: Assign a priority to each filter. If a filter is specific, mark it "required": true.
5. Search Variants: Generate a "semantic_query" (for Vector Search) and a "keyword_string" (for BM25).

User Input: "{message}"

Lịch sử hội thoại:
{history}

Output Format (Strict JSON):
{
  "semantic_query": "Expanded descriptive sentence...",
  "keyword_string": "Optimized keywords...",
  "filters": {
    "min_experience": {"value": 0, "required": false},
    "location": {"value": null, "required": false},
    "skills": []
  },
  "explanation": "Briefly explain...",
  "is_search_needed": true,
  "intent": "search | list_all | compare | chat | off_topic"
}

Intent values:
- "search": User wants to find candidates with SPECIFIC criteria (skills, location, experience)
- "list_all": User wants to see ALL candidates or browse without specific criteria (e.g., "Liệt kê ứng viên", "Cho tôi xem CV", "Có bao nhiêu ứng viên?")
- "chat": User is having general conversation about recruiting
- "off_topic": User is asking about something NOT related to recruitment/CVs

IMPORTANT: If user asks to "list", "show", "browse", "xem", "liệt kê" candidates WITHOUT specific skills/location, use "list_all" intent.
"""


class QueryTransformer:
    """
    Transforms user messages into structured search queries and filters.
    
    Uses a small LLM to analyze the user's intent and extract:
    - Optimized search query for vector search
    - Structured filters for SQL/metadata filtering
    - Intent classification
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize query transformer.
        
        Args:
            api_key: Groq API key
            model: Model to use for transformation (smaller is better for speed)
        """
        self.api_key = api_key or settings.groq_api_key
        # Use a smaller, faster model for query transformation
        self.model = model or "llama-3.1-8b-instant"
        self._client: Optional[Groq] = None
    
    def _get_client(self) -> Groq:
        """Get or create Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    def transform(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
    ) -> TransformedQuery:
        """
        Transform a user message into a structured query.
        
        Args:
            message: Current user message
            history: Previous conversation messages
            
        Returns:
            TransformedQuery with search_query, filters, and intent
        """
        try:
            # Format history for prompt
            history_str = ""
            if history:
                history_lines = []
                for msg in history[-5:]:  # Last 5 messages for context
                    role = "User" if msg.role.value == "user" else "Assistant"
                    history_lines.append(f"{role}: {msg.content[:200]}")
                history_str = "\n".join(history_lines)
            
            prompt = QUERY_TRANSFORM_PROMPT.replace(
                "{history}", history_str or "(Không có lịch sử)"
            ).replace(
                "{message}", message
            )
            
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON-only response bot."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"Raw Query Transform Response: {result_text}")
            
            # Extract JSON object using regex (most robust way)
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            result = json.loads(result_text)
            
            # Map new JSON structure to internal schema
            filters = result.get("filters", {})
            semantic_query = result.get("semantic_query", message)
            keyword_string = result.get("keyword_string", message)
            
            # Flatten filters for internal use
            flattened_filters = {}
            if filters.get("location") and filters["location"].get("value"):
                flattened_filters["location"] = filters["location"]["value"]
            
            if filters.get("min_experience") and filters["min_experience"].get("value"):
                # Handle possible string/int types
                try:
                    flattened_filters["min_experience_years"] = float(filters["min_experience"]["value"])
                except (ValueError, TypeError):
                    pass
            
            if filters.get("skills"):
                skills_data = filters["skills"]
                # Handle list of strings or list of objects (case where LLM changes structure)
                flattened_skills = []
                for skill in skills_data:
                    if isinstance(skill, str):
                        flattened_skills.append(skill)
                    elif isinstance(skill, dict) and "name" in skill:
                        flattened_skills.append(skill["name"])
                
                if flattened_skills:
                    flattened_filters["required_skills"] = flattened_skills

            return TransformedQuery(
                search_query=semantic_query, # Legacy mapping
                semantic_query=semantic_query,
                keyword_string=keyword_string,
                filters=flattened_filters,
                is_search_needed=result.get("is_search_needed", True),
                intent=result.get("intent", "search"),
                explanation=result.get("explanation"),
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse query transform response: {e}")
            # Fallback: use original message
            return TransformedQuery(
                search_query=message,
                semantic_query=message,
                keyword_string=message,
                filters={},
                is_search_needed=True,
                intent="search",
            )
        except Exception as e:
            logger.exception(f"Query transformation failed: {e}")
            # Fallback
            return TransformedQuery(
                search_query=message,
                semantic_query=message,
                keyword_string=message,
                filters={},
                is_search_needed=True,
                intent="search",
            )
    
    async def transform_async(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
    ) -> TransformedQuery:
        """Async wrapper for transform (runs sync in executor)."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transform(message, history)
        )


# Singleton instance
_transformer: Optional[QueryTransformer] = None


def get_query_transformer() -> QueryTransformer:
    """Get or create the query transformer singleton."""
    global _transformer
    if _transformer is None:
        _transformer = QueryTransformer()
    return _transformer
