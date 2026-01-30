"""
Response Critic - Evaluates chatbot responses before returning to user.

Implements a quality gate with retry loop:
1. Generate response from RAG chain
2. Evaluate response quality
3. If score < threshold, regenerate with feedback
4. Maximum 2 retries before returning best response
"""

import json
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from groq import Groq

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class CriticResult:
    """Result of response evaluation."""
    score: float  # 1-10
    feedback: str
    relevance_score: float
    accuracy_score: float
    formatting_score: float
    completeness_score: float
    should_regenerate: bool
    improvement_hints: list


CRITIC_PROMPT = """You are a Response Quality Critic for a CV screening chatbot. Evaluate this response.

USER QUERY:
{query}

CONTEXT (CV data available):
{context}

CHATBOT RESPONSE:
{response}

Evaluate from 1-10 on these criteria:
1. **Relevance** (25%): Does the response directly answer the user's question?
2. **Accuracy** (30%): Is the response factually grounded in the context data? Does it avoid hallucination?
3. **Formatting** (20%): Is it well-formatted with Markdown (tables, headings, bullet points)?
4. **Completeness** (25%): Does it cover all aspects of the query without missing important details?

Return ONLY valid JSON:
{{
  "overall_score": 7.5,
  "relevance_score": 8.0,
  "accuracy_score": 7.0,
  "formatting_score": 8.0,
  "completeness_score": 7.0,
  "feedback": "Brief summary of evaluation",
  "should_regenerate": true,
  "improvement_hints": ["Be more specific about X", "Include table for comparison"]
}}"""


REGENERATION_SYSTEM_PROMPT = """You are a helpful CV screening assistant. Your previous response was not optimal.

FEEDBACK FROM CRITIC:
{feedback}

IMPROVEMENT HINTS:
{hints}

Please regenerate a better response that addresses these issues while:
1. Staying factually grounded in the provided context
2. Using proper Markdown formatting (tables, headings, bullet points)
3. Being comprehensive but concise
4. Directly answering the user's question"""


class ResponseCritic:
    """
    Evaluates chatbot responses before returning to user.
    
    Uses a smaller model for evaluation to reduce latency and cost.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        critic_model: str = "llama-3.1-8b-instant",
        min_score: float = 8.0,
        max_retries: int = 2,
    ):
        """
        Initialize the critic.
        
        Args:
            api_key: Groq API key
            critic_model: Model for evaluation (smaller/faster)
            min_score: Minimum score to pass without regeneration
            max_retries: Maximum regeneration attempts
        """
        self._api_key = api_key or settings.groq_api_key
        self._critic_model = critic_model
        self._min_score = min_score
        self._max_retries = max_retries
        self._client: Optional[Groq] = None
        
    def _get_client(self) -> Groq:
        """Get or create Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client
    
    def evaluate(
        self,
        query: str,
        response: str,
        context: str,
    ) -> CriticResult:
        """
        Evaluate a chatbot response.
        
        Args:
            query: User's original query
            response: Generated response to evaluate
            context: CV context data used to generate response
            
        Returns:
            CriticResult with scores and feedback
        """
        client = self._get_client()
        
        prompt = CRITIC_PROMPT.format(
            query=query,
            context=context[:4000],  # Limit context size
            response=response[:2000],  # Limit response size
        )
        
        try:
            result = client.chat.completions.create(
                model=self._critic_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            data = json.loads(result.choices[0].message.content)
            
            return CriticResult(
                score=data.get("overall_score", 7.0),
                feedback=data.get("feedback", ""),
                relevance_score=data.get("relevance_score", 7.0),
                accuracy_score=data.get("accuracy_score", 7.0),
                formatting_score=data.get("formatting_score", 7.0),
                completeness_score=data.get("completeness_score", 7.0),
                should_regenerate=data.get("should_regenerate", False),
                improvement_hints=data.get("improvement_hints", []),
            )
            
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            # Return passing score on error
            return CriticResult(
                score=8.0,
                feedback=f"Evaluation error: {e}",
                relevance_score=8.0,
                accuracy_score=8.0,
                formatting_score=8.0,
                completeness_score=8.0,
                should_regenerate=False,
                improvement_hints=[],
            )
    
    def should_retry(self, result: CriticResult, attempt: int) -> bool:
        """
        Determine if regeneration is needed.
        
        Args:
            result: Evaluation result
            attempt: Current attempt number (0-indexed)
            
        Returns:
            True if should regenerate
        """
        if attempt >= self._max_retries:
            return False
        return result.score < self._min_score and result.should_regenerate
    
    def get_regeneration_prompt(self, result: CriticResult) -> str:
        """
        Build the system prompt for regeneration.
        
        Args:
            result: Critic evaluation result
            
        Returns:
            System prompt with feedback
        """
        hints = "\n".join(f"- {hint}" for hint in result.improvement_hints)
        
        return REGENERATION_SYSTEM_PROMPT.format(
            feedback=result.feedback,
            hints=hints or "- Improve overall quality",
        )


# Singleton instance
_critic: Optional[ResponseCritic] = None


def get_response_critic() -> ResponseCritic:
    """Get or create the response critic singleton."""
    global _critic
    if _critic is None:
        _critic = ResponseCritic()
    return _critic
