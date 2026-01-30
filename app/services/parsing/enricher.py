"""
Contextual Enricher for CV chunks.

Before embedding any chunk, we prepend contextual metadata to help
the vector search distinguish between similar content in different contexts.

Example:
    Original: "Led a team of 5 engineers to deliver a mobile app"
    Enriched: "Nguyen Van A - Senior Developer - FPT Software - 2020-2023
              Led a team of 5 engineers to deliver a mobile app"

This enrichment helps the embedding model understand:
- WHO did this (candidate name)
- WHAT role they had (job title)
- WHERE they worked (company)
- WHEN they did this (duration)
"""

import logging
from typing import Optional, List, Dict, Any

from app.services.parsing.chunker import Chunk, SectionType
from app.schemas.resume import ResumeSchema

logger = logging.getLogger(__name__)


class ContextualEnricher:
    """
    Enriches CV chunks with contextual metadata for better embedding.
    
    The enriched content becomes the text that gets embedded, ensuring
    that semantic search can differentiate between similar experiences
    from different candidates or at different seniority levels.
    """

    def __init__(
        self,
        include_name: bool = True,
        include_position: bool = True,
        include_company: bool = True,
        include_duration: bool = True,
        separator: str = " | ",
    ):
        """
        Configure what context to include.
        
        Args:
            include_name: Include candidate name in context
            include_position: Include job position/title
            include_company: Include company/organization name
            include_duration: Include time period
            separator: String to separate context parts
        """
        self.include_name = include_name
        self.include_position = include_position
        self.include_company = include_company
        self.include_duration = include_duration
        self.separator = separator

    def enrich_chunk(
        self,
        chunk: Chunk,
        resume: Optional[ResumeSchema] = None,
        candidate_name: Optional[str] = None,
    ) -> str:
        """
        Enrich a single chunk with contextual metadata.
        
        Args:
            chunk: The chunk to enrich
            resume: Full resume schema (provides candidate name)
            candidate_name: Override candidate name
            
        Returns:
            Enriched text ready for embedding
        """
        # Get candidate name
        name = candidate_name or (resume.full_name if resume else None)

        # Build context string from chunk metadata
        context_parts = []

        if self.include_name and name:
            context_parts.append(name)

        metadata = chunk.metadata or {}

        if self.include_position and "position" in metadata:
            context_parts.append(metadata["position"])

        if self.include_company and "company" in metadata:
            context_parts.append(metadata["company"])

        if self.include_duration and "duration" in metadata:
            context_parts.append(metadata["duration"])

        # Add section-specific context
        section_context = self._get_section_context(chunk)
        if section_context:
            context_parts.append(section_context)

        # Build enriched content
        if context_parts:
            context = self.separator.join(context_parts)
            enriched = f"{context}\n\n{chunk.content}"
        else:
            enriched = chunk.content

        return enriched

    def enrich_chunks(
        self,
        chunks: List[Chunk],
        resume: Optional[ResumeSchema] = None,
    ) -> List[Chunk]:
        """
        Enrich multiple chunks and update their enriched_content.
        
        This modifies chunks in place and also returns them.
        
        Args:
            chunks: List of chunks to enrich
            resume: Resume schema for context
            
        Returns:
            List of chunks with enriched_content populated
        """
        candidate_name = resume.full_name if resume else None

        for chunk in chunks:
            # Update metadata with resume info if available
            if resume and candidate_name:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["candidate_name"] = candidate_name

            # Generate enriched content
            enriched = self.enrich_chunk(chunk, resume, candidate_name)

            # Store back (we'll use a dict since Chunk is a dataclass)
            if hasattr(chunk, "enriched_content"):
                # This is for when we're returning it
                pass

            # Add enriched content to metadata for later use
            chunk.metadata["enriched_content"] = enriched

        return chunks

    def _get_section_context(self, chunk: Chunk) -> Optional[str]:
        """Get section-specific context string."""
        section = chunk.section

        if section == SectionType.EXPERIENCE:
            return "Work Experience"
        elif section == SectionType.EDUCATION:
            return "Education"
        elif section == SectionType.PROJECTS:
            return "Project"
        elif section == SectionType.SKILLS:
            return "Skills"
        elif section == SectionType.CERTIFICATIONS:
            return "Certification"

        return None

    def enrich_from_work_experience(
        self,
        content: str,
        candidate_name: str,
        position: str,
        company: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Create enriched content for a work experience entry.
        
        This is a convenience method when you have structured data
        instead of a Chunk object.
        
        Args:
            content: Work experience description
            candidate_name: Name of the candidate
            position: Job title
            company: Company name
            start_date: Start date string
            end_date: End date string (None for current position)
            
        Returns:
            Enriched text for embedding
        """
        parts = [candidate_name, position, company]

        if end_date:
            parts.append(f"{start_date} - {end_date}")
        else:
            parts.append(f"{start_date} - Present")

        context = self.separator.join(parts)
        return f"{context}\n\n{content}"

    def enrich_from_education(
        self,
        content: str,
        candidate_name: str,
        degree: str,
        institution: str,
        graduation_year: Optional[str] = None,
    ) -> str:
        """
        Create enriched content for an education entry.
        
        Args:
            content: Education description
            candidate_name: Name of the candidate
            degree: Degree type and field
            institution: School/university name
            graduation_year: Year of graduation
            
        Returns:
            Enriched text for embedding
        """
        parts = [candidate_name, degree, institution]

        if graduation_year:
            parts.append(graduation_year)

        context = self.separator.join(parts)
        return f"{context}\n\n{content}"

    def enrich_from_project(
        self,
        content: str,
        candidate_name: str,
        project_name: str,
        role: Optional[str] = None,
        technologies: Optional[List[str]] = None,
    ) -> str:
        """
        Create enriched content for a project entry.
        
        Args:
            content: Project description
            candidate_name: Name of the candidate
            project_name: Name of the project
            role: Role in the project
            technologies: Technologies used
            
        Returns:
            Enriched text for embedding
        """
        parts = [candidate_name, "Project", project_name]

        if role:
            parts.append(role)

        if technologies:
            parts.append(f"Technologies: {', '.join(technologies[:5])}")

        context = self.separator.join(parts)
        return f"{context}\n\n{content}"

    def build_summary_context(self, resume: ResumeSchema) -> str:
        """
        Build a rich context string for the resume summary embedding.
        
        This creates a comprehensive context that includes:
        - Candidate name and headline
        - Current/most recent position
        - Top skills
        - Total experience
        
        Args:
            resume: Complete resume schema
            
        Returns:
            Context string for summary embedding
        """
        parts = [resume.full_name]

        if resume.headline:
            parts.append(resume.headline)

        # Add most recent position
        if resume.work_experience:
            recent = resume.work_experience[0]
            parts.append(f"{recent.position} at {recent.company}")

        # Add top skills
        skills = resume.get_all_skills()[:5]
        if skills:
            parts.append(f"Skills: {', '.join(skills)}")

        context = self.separator.join(parts)

        if resume.summary:
            return f"{context}\n\n{resume.summary}"
        else:
            return context


# Singleton instance
_enricher: Optional[ContextualEnricher] = None


def get_enricher() -> ContextualEnricher:
    """Get or create the contextual enricher singleton."""
    global _enricher
    if _enricher is None:
        _enricher = ContextualEnricher()
    return _enricher
