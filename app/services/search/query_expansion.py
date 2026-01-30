"""
Query Expansion using LLM.

When a user searches for "AI Engineer", the system expands the query into
multiple variations like:
- "Machine Learning Engineer"
- "NLP Scientist" 
- "Kỹ sư trí tuệ nhân tạo" (Vietnamese)
- "Deep Learning Specialist"

This improves recall by matching candidates who may use different
terminology for similar roles/skills.
"""

import json
import logging
from typing import List, Optional

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# Prompt for query expansion
EXPANSION_PROMPT = """You are a recruitment search expert. Your task is to expand a search query into related terms, synonyms, and translations that would match relevant candidates.

RULES:
1. Generate 3-5 alternative queries
2. Include both English and Vietnamese variations if applicable
3. Include common synonyms and related job titles/skills
4. Keep variations relevant and not too broad
5. Return as a JSON array of strings

EXAMPLES:
Query: "AI Engineer"
Output: ["Machine Learning Engineer", "NLP Scientist", "Deep Learning Engineer", "Kỹ sư trí tuệ nhân tạo", "Data Scientist AI"]

Query: "Python developer"
Output: ["Python programmer", "Backend Python developer", "Lập trình viên Python", "Python software engineer"]

Query: "Project management"
Output: ["Project manager", "PM", "Quản lý dự án", "Program manager", "Scrum master"]

Now expand the following query:"""


class QueryExpander:
    """
    Expands search queries using LLM for better search coverage.
    
    Uses GPT-4o-mini for cost-effective query expansion.
    Falls back to simple variations if LLM is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize query expander.
        
        Args:
            api_key: OpenAI API key (defaults to settings)
        """
        self._client = None
        self._api_key = api_key or settings.openai_api_key
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of OpenAI client."""
        if self._initialized:
            return

        if not self._api_key:
            logger.warning(
                "OpenAI API key not set. Query expansion will use fallback."
            )
            self._initialized = True
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            self._initialized = True
        except ImportError:
            logger.warning("OpenAI not installed. Using fallback expansion.")
            self._initialized = True

    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand a search query into related variations.
        
        Args:
            query: Original search query
            max_expansions: Maximum number of expansions
            
        Returns:
            List of query variations (including original)
        """
        self._lazy_init()

        # Always include original query
        result = [query]

        if self._client:
            try:
                expansions = self._expand_with_llm(query)
                result.extend(expansions[:max_expansions])
            except Exception as e:
                logger.warning(f"LLM expansion failed: {e}")
                result.extend(self._fallback_expansion(query))
        else:
            result.extend(self._fallback_expansion(query))

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for q in result:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q.strip())

        return unique[:max_expansions + 1]

    def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM."""
        prompt = f"{EXPANSION_PROMPT}\nQuery: \"{query}\""

        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Handle case where response might have markdown code block
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            expansions = json.loads(content)

            if isinstance(expansions, list):
                return [str(e) for e in expansions if e]
        except json.JSONDecodeError:
            # Try to extract strings from response
            import re
            strings = re.findall(r'"([^"]+)"', content)
            if strings:
                return strings

        return []

    def _fallback_expansion(self, query: str) -> List[str]:
        """
        Simple fallback expansion without LLM.
        
        Uses basic rules for common job titles and skills.
        """
        expansions = []
        query_lower = query.lower()

        # Common expansions for job titles
        title_mappings = {
            "developer": ["engineer", "programmer", "lập trình viên"],
            "engineer": ["developer", "specialist", "kỹ sư"],
            "manager": ["lead", "director", "quản lý"],
            "senior": ["sr.", "experienced"],
            "junior": ["jr.", "entry-level", "fresher"],
        }

        # Check for matching patterns
        for key, expansions_list in title_mappings.items():
            if key in query_lower:
                for exp in expansions_list:
                    new_query = query_lower.replace(key, exp)
                    expansions.append(new_query.title())

        # Add Vietnamese translation patterns
        vn_mappings = {
            "software": "phần mềm",
            "developer": "lập trình viên",
            "engineer": "kỹ sư",
            "manager": "quản lý",
            "data": "dữ liệu",
            "web": "web",
            "mobile": "di động",
        }

        # Try to create Vietnamese version
        vn_query = query
        for en, vn in vn_mappings.items():
            vn_query = vn_query.lower().replace(en, vn)
        if vn_query.lower() != query.lower():
            expansions.append(vn_query.title())

        return expansions

    def expand_for_skills(self, skill: str) -> List[str]:
        """
        Expand a skill into related skills.
        
        Used for skill matching in candidate search.
        
        Args:
            skill: Skill to expand (e.g., "Python")
            
        Returns:
            Related skills
        """
        # Common skill groupings
        skill_groups = {
            "python": ["python3", "django", "flask", "fastapi"],
            "javascript": ["js", "node.js", "react", "vue", "angular"],
            "java": ["spring", "spring boot", "hibernate"],
            "sql": ["mysql", "postgresql", "oracle", "sql server"],
            "aws": ["amazon web services", "ec2", "s3", "lambda"],
            "docker": ["container", "kubernetes", "k8s"],
            "machine learning": ["ml", "deep learning", "ai", "học máy"],
        }

        skill_lower = skill.lower()

        for key, related in skill_groups.items():
            if skill_lower == key or skill_lower in related:
                return [skill] + [s for s in related if s != skill_lower]

        return [skill]


# Singleton instance
_expander: Optional[QueryExpander] = None


def get_query_expander() -> QueryExpander:
    """Get or create the query expander singleton."""
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander
