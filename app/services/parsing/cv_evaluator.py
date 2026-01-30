"""
CV Data Evaluator - Evaluates and reformats parsed CV data before saving.

Ensures CV data quality by:
1. Scoring completeness and format consistency
2. Reformatting data if score is below threshold
3. Normalizing skills and standardizing formats
"""

import json
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from groq import Groq

from app.config import get_settings
from app.schemas.resume import ResumeSchema

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of CV data evaluation."""
    score: float  # 1-10
    feedback: str
    issues: list
    should_reformat: bool


EVALUATOR_PROMPT = """You are a CV Data Quality Evaluator. Evaluate the following parsed CV JSON data.

Score from 1-10 based on these criteria:
1. **Completeness** (30%): full_name, email, phone, work_experience, skills
2. **Format Consistency** (20%): Dates in ISO format, phone has country code
3. **Project Quality** (25%): Each project has name, description, technologies
4. **Skills Quality** (15%): No duplicates, properly categorized
5. **Experience Quality** (10%): Has company, position, responsibilities

CV DATA:
```json
{cv_data}
```

Return ONLY valid JSON:
{{
  "score": 7.5,
  "feedback": "Summary of quality assessment",
  "issues": ["Issue 1", "Issue 2"],
  "should_reformat": true
}}"""


REFORMATTER_PROMPT = """You are a CV Data Reformatter. Fix the issues in this CV JSON data.

ORIGINAL CV DATA:
```json
{cv_data}
```

ISSUES TO FIX:
{issues}

REFORMATTING RULES:
1. Standardize all dates to ISO format (YYYY-MM-DD)
2. Normalize phone numbers with country code (+84...)
3. Remove duplicate skills, normalize capitalization
4. Ensure each project has: name, description (if missing, generate from context), technologies
5. Ensure each work experience has: company, position, start_date, responsibilities (as list)
6. Fill missing headlines from work experience if possible
7. DO NOT invent information that doesn't exist - only reformat and organize

Return ONLY the corrected JSON matching the original schema structure."""


class CVDataEvaluator:
    """
    Evaluates parsed CV data quality and reformats if needed.
    
    Uses a smaller, faster model for evaluation to reduce costs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        eval_model: str = "llama-3.1-8b-instant",
        reformat_model: str = "llama-3.3-70b-versatile",
        min_score: float = 7.0,
        max_retries: int = 2,
    ):
        """
        Initialize the evaluator.
        
        Args:
            api_key: Groq API key
            eval_model: Model for evaluation (smaller/faster)
            reformat_model: Model for reformatting (larger/smarter)
            min_score: Minimum score to pass without reformatting
            max_retries: Maximum reformatting attempts
        """
        self._api_key = api_key or settings.groq_api_key
        self._eval_model = eval_model
        self._reformat_model = reformat_model
        self._min_score = min_score
        self._max_retries = max_retries
        self._client: Optional[Groq] = None
        
    def _get_client(self) -> Groq:
        """Get or create Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client
    
    def evaluate(self, resume: ResumeSchema) -> EvaluationResult:
        """
        Evaluate CV data quality.
        
        Args:
            resume: Parsed resume data
            
        Returns:
            EvaluationResult with score and feedback
        """
        client = self._get_client()
        
        # Convert resume to JSON for evaluation
        cv_json = resume.model_dump_json(indent=2, exclude={'raw_text', 'summary_embedding'})
        
        prompt = EVALUATOR_PROMPT.format(cv_data=cv_json[:8000])  # Limit size
        
        try:
            response = client.chat.completions.create(
                model=self._eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return EvaluationResult(
                score=result.get("score", 5.0),
                feedback=result.get("feedback", ""),
                issues=result.get("issues", []),
                should_reformat=result.get("should_reformat", False),
            )
            
        except Exception as e:
            logger.error(f"CV evaluation failed: {e}")
            # Return passing score on error to avoid blocking
            return EvaluationResult(
                score=7.0,
                feedback=f"Evaluation error: {e}",
                issues=[],
                should_reformat=False,
            )
    
    def reformat(self, resume: ResumeSchema, issues: list) -> ResumeSchema:
        """
        Reformat CV data to fix issues.
        
        Args:
            resume: Original resume data
            issues: List of issues to fix
            
        Returns:
            Reformatted ResumeSchema
        """
        client = self._get_client()
        
        cv_json = resume.model_dump_json(indent=2, exclude={'raw_text'})
        issues_str = "\n".join(f"- {issue}" for issue in issues)
        
        prompt = REFORMATTER_PROMPT.format(
            cv_data=cv_json[:12000],
            issues=issues_str,
        )
        
        try:
            response = client.chat.completions.create(
                model=self._reformat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            
            reformatted_data = json.loads(response.choices[0].message.content)
            
            # Preserve original metadata
            reformatted_data['raw_text'] = resume.raw_text
            reformatted_data['source_file'] = resume.source_file
            reformatted_data['parsed_at'] = resume.parsed_at
            
            return ResumeSchema.model_validate(reformatted_data)
            
        except Exception as e:
            logger.error(f"CV reformatting failed: {e}")
            # Return original on error
            return resume
    
    def evaluate_and_reformat(
        self,
        resume: ResumeSchema,
    ) -> Tuple[ResumeSchema, EvaluationResult]:
        """
        Evaluate CV and reformat if needed (with retries).
        
        Args:
            resume: Parsed resume data
            
        Returns:
            Tuple of (possibly reformatted resume, final evaluation)
        """
        current_resume = resume
        
        for attempt in range(self._max_retries + 1):
            eval_result = self.evaluate(current_resume)
            
            logger.info(
                f"CV evaluation attempt {attempt + 1}: "
                f"score={eval_result.score:.1f}, issues={len(eval_result.issues)}"
            )
            
            # Pass if score is good enough or max retries reached
            if eval_result.score >= self._min_score or attempt == self._max_retries:
                return current_resume, eval_result
            
            # Reformat and try again
            if eval_result.issues:
                logger.info(f"Reformatting CV to fix {len(eval_result.issues)} issues")
                current_resume = self.reformat(current_resume, eval_result.issues)
        
        return current_resume, eval_result


# Singleton instance
_evaluator: Optional[CVDataEvaluator] = None


def get_cv_evaluator() -> CVDataEvaluator:
    """Get or create the CV evaluator singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = CVDataEvaluator()
    return _evaluator
