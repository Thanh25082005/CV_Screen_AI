"""
LLM Parser using Groq or OpenAI for CV parsing.

This module handles:
1. Extracting structured data from CV text using LLM
2. JSON mode for reliable output format
3. Validation of extracted data
4. Automatic flagging of missing/invalid information
5. Support for multiple LLM providers (Groq, OpenAI)
"""

import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.config import get_settings
from app.schemas.resume import ResumeSchema

settings = get_settings()
logger = logging.getLogger(__name__)


# System prompt for CV parsing
SYSTEM_PROMPT = """You are an expert CV/Resume parser. Your task is to extract structured information from CV text.

IMPORTANT GUIDELINES:
1. Extract ALL available information from the CV
2. For dates, use ISO format (YYYY-MM-DD). If only year is available, use YYYY-01-01
3. For missing end dates in current positions, use null
4. Preserve original language for names, companies, and descriptions
5. For skills, extract both technical skills (Python, AWS) and soft skills (Leadership, Communication)
6. If information is unclear or missing, use null rather than guessing

LANGUAGE HANDLING:
- The CV may be in Vietnamese, English, or mixed
- Preserve the original language of content
- Recognize Vietnamese date formats (e.g., "01/2020 - 12/2023")
- Handle Vietnamese job titles and company names

PROJECT EXTRACTION RULES (CRITICAL):
1. Search for keywords: "Dự án", "Project", "Đề tài", "Luận văn", "Thesis", "Side Project", "Personal Project", "Capstone"
2. Projects may appear INSIDE Work Experience entries as sub-items or bullet points - EXTRACT THEM
3. Look for project mentions in these patterns:
   - "Tham gia dự án X" / "Participated in project X"
   - "Phụ trách dự án Y" / "Led project Y"  
   - "Xây dựng hệ thống Z" / "Built system Z"
   - "Phát triển ứng dụng W" / "Developed application W"
4. If no "Projects" section exists, search for project names in:
   - Work Experience descriptions and achievements
   - Awards/Achievements (e.g., "Giải nhất dự án ABC")
   - Education section (thesis, capstone, final year project)
5. Each project MUST have a name. Use the most descriptive title found.
6. Extract technologies from project descriptions (frameworks, languages, tools mentioned)

OUTPUT REQUIREMENTS:
- Return valid JSON matching the provided schema
- All dates in ISO format or null
- All lists as arrays (even if empty)
- Email must be valid format or null
- Phone numbers should include country code if available
- Projects array should NEVER be empty if any project is mentioned anywhere in the CV"""


class LLMParser:
    """
    LLM-based CV parser supporting multiple providers (Groq, OpenAI).
    
    Uses JSON mode for reliable output that conforms to the ResumeSchema.
    """

    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize the LLM parser.
        
        Args:
            api_key: API key (defaults to settings based on provider)
            provider: LLM provider ('groq' or 'openai', defaults to settings)
        """
        self._client = None
        self._provider = provider or settings.llm_provider
        self._initialized = False
        
        # Set API key based on provider
        if api_key:
            self._api_key = api_key
        elif self._provider == "groq":
            self._api_key = settings.groq_api_key
        else:
            self._api_key = settings.openai_api_key

    def _lazy_init(self):
        """Lazy initialization of LLM client."""
        if self._initialized:
            return

        if not self._api_key:
            raise ValueError(
                f"{self._provider.upper()} API key not provided. "
                f"Set {'GROQ_API_KEY' if self._provider == 'groq' else 'OPENAI_API_KEY'} environment variable."
            )

        try:
            if self._provider == "groq":
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
                self._model = settings.groq_model
                logger.info(f"Groq client initialized with model: {self._model}")
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
                self._model = "gpt-4o"
                logger.info("OpenAI client initialized for LLM parsing")
            
            self._initialized = True
            
        except ImportError as e:
            pkg = "groq" if self._provider == "groq" else "openai"
            raise ImportError(f"{pkg} package not installed. Run: pip install {pkg}")

    def parse_resume(
        self,
        text: str,
        source_filename: Optional[str] = None,
    ) -> ResumeSchema:
        """
        Parse CV text and extract structured data.
        
        Args:
            text: Raw CV text to parse
            source_filename: Original filename for metadata
            
        Returns:
            ResumeSchema with extracted data and validation warnings
        """
        self._lazy_init()

        if not text or len(text.strip()) < 50:
            raise ValueError("CV text is too short or empty")

        return self._parse_with_json_mode(text, source_filename)

    def parse_resume_with_fallback(
        self,
        text: str,
        source_filename: Optional[str] = None,
    ) -> ResumeSchema:
        """
        Parse CV with JSON mode.
        """
        return self.parse_resume(text, source_filename)

    def _parse_with_json_mode(
        self,
        text: str,
        source_filename: Optional[str] = None,
    ) -> ResumeSchema:
        """
        Parse CV using JSON mode for structured output.
        Includes retry logic for rate limit errors.
        """
        import time
        
        self._lazy_init()

        # Create a JSON schema from Pydantic model
        schema = ResumeSchema.model_json_schema()

        user_prompt = f"""Parse the following CV/Resume and return a JSON object.

JSON Schema:
{json.dumps(schema, indent=2)}

---CV TEXT START---
{text[:15000]}
---CV TEXT END---

Return ONLY valid JSON matching the schema above, no other text."""

        max_retries = 3
        base_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                if self._provider == "groq":
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=8192,
                        response_format={"type": "json_object"},
                    )
                else:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )

                # Parse JSON response
                json_str = response.choices[0].message.content
                data = json.loads(json_str)

                # Validate with Pydantic
                resume = ResumeSchema.model_validate(data)
                resume.source_file = source_filename
                resume.parsed_at = datetime.utcnow().isoformat()
                resume.raw_text = text

                resume = self._validate_and_flag(resume)

                logger.info(
                    f"Successfully parsed CV: {resume.full_name}, "
                    f"{len(resume.work_experience)} jobs, "
                    f"{len(resume.skills)} skills, "
                    f"{len(resume.validation_warnings)} warnings"
                )

                return resume

            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Rate limit hit, waiting {delay}s before retry "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                
                logger.error(f"Failed to parse CV with LLM: {e}")
                raise

    def _validate_and_flag(self, resume: ResumeSchema) -> ResumeSchema:
        """
        Add validation warnings for missing or suspicious data.
        """
        warnings = list(resume.validation_warnings)

        # Check for obviously fake or placeholder data
        if resume.email and "example.com" in resume.email:
            warnings.append("Suspicious email: appears to be placeholder")

        if resume.phone and len(resume.phone.replace("+", "").replace("-", "")) < 8:
            warnings.append("Phone number appears too short")

        # Check work experience
        for i, exp in enumerate(resume.work_experience):
            if exp.start_date and exp.end_date:
                if exp.start_date > exp.end_date:
                    warnings.append(
                        f"Work Experience {i+1}: End date is before start date"
                    )
                elif (exp.end_date - exp.start_date).days > 365 * 20:
                    warnings.append(
                        f"Work Experience {i+1}: Duration seems unusually long"
                    )

        # Check education
        for edu in resume.education:
            if edu.gpa and edu.gpa > 4.0:
                warnings.append(
                    f"Education at {edu.institution}: GPA > 4.0, may need review"
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_warnings = []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                unique_warnings.append(w)

        resume.validation_warnings = unique_warnings
        return resume

    def extract_specific_field(
        self,
        text: str,
        field: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract a specific field from CV text using LLM.
        """
        self._lazy_init()

        prompt = f"""Extract the {field} from this CV text.
        
{context or ""}

CV Text:
{text[:5000]}

Return ONLY the extracted value, nothing else. If not found, return "NOT_FOUND"."""

        if self._provider == "groq":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )
        else:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )

        result = response.choices[0].message.content.strip()

        if result == "NOT_FOUND":
            return None
        return result


# Singleton instance
_parser: Optional[LLMParser] = None


def get_parser() -> LLMParser:
    """Get or create the LLM parser singleton."""
    global _parser
    if _parser is None:
        _parser = LLMParser()
    return _parser
