"""
Search and Matching API Routes.

Endpoints for:
- Hybrid search (BM25 + Vector + RRF)
- Job-to-candidate matching
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_async_db, get_search_engine_dep
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    MatchRequest,
    MatchResponse,
    MatchResult,
    MatchScore,
)
from app.services.search.hybrid import HybridSearchEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search & Matching"])


@router.post("", response_model=SearchResponse)
async def search_candidates(
    request: SearchRequest,
    db: AsyncSession = Depends(get_async_db),
    search_engine: HybridSearchEngine = Depends(get_search_engine_dep),
):
    """
    Search for candidates using hybrid search.
    
    Combines:
    - BM25 keyword search for exact skill matching
    - Vector semantic search for meaning-based matching
    - RRF fusion for optimal ranking
    
    Supports query expansion to find related terms.
    """
    try:
        response = await search_engine.search(request, db)

        # Enrich results with candidate details
        if response.results:
            await _enrich_search_results(response.results, db)

        return response

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.post("/match", response_model=MatchResponse)
async def match_candidates(
    request: MatchRequest,
    db: AsyncSession = Depends(get_async_db),
    search_engine: HybridSearchEngine = Depends(get_search_engine_dep),
):
    """
    Match candidates against a job description.
    
    Uses the job description to:
    1. Find relevant candidates
    2. Score them on skill match, experience, and semantic fit
    3. Return ranked matches with explanations
    """
    start_time = time.time()

    try:
        # Create search request from match request
        search_request = SearchRequest(
            query=request.job_description,
            top_k=request.top_k * 2,  # Get more candidates for filtering
            expand_query=True,
            required_skills=request.required_skills,
        )

        # Execute search
        search_response = await search_engine.search(search_request, db)

        if not search_response.results:
            return MatchResponse(
                job_title=request.job_title,
                total_candidates_evaluated=0,
                total_matches=0,
                matches=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Enrich and score candidates
        matches = await _score_candidates_for_job(
            search_response.results,
            request,
            db,
        )

        # Filter and sort by overall score
        matches = [m for m in matches if m.match_score.overall_score >= 30]
        matches.sort(key=lambda x: x.match_score.overall_score, reverse=True)
        matches = matches[:request.top_k]

        return MatchResponse(
            job_title=request.job_title,
            total_candidates_evaluated=len(search_response.results),
            total_matches=len(matches),
            matches=matches,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Matching failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}",
        )


async def _enrich_search_results(
    results,
    db: AsyncSession,
) -> None:
    """Enrich search results with candidate details from DB."""
    from sqlalchemy import select
    from app.models.candidate import Candidate

    candidate_ids = [r.candidate_id for r in results]

    query = select(Candidate).where(Candidate.id.in_(candidate_ids))
    result = await db.execute(query)
    candidates = {c.id: c for c in result.scalars().all()}

    for r in results:
        if r.candidate_id in candidates:
            c = candidates[r.candidate_id]
            r.full_name = c.full_name
            r.email = c.email
            r.headline = c.headline
            r.total_experience_years = c.total_experience_years
            r.top_skills = c.top_skills or []


async def _score_candidates_for_job(
    search_results,
    request: MatchRequest,
    db: AsyncSession,
) -> list:
    """Score candidates against job requirements."""
    from sqlalchemy import select
    from app.models.candidate import Candidate

    candidate_ids = [r.candidate_id for r in search_results]

    query = select(Candidate).where(Candidate.id.in_(candidate_ids))
    result = await db.execute(query)
    candidates = {c.id: c for c in result.scalars().all()}

    matches = []
    required_skills_lower = set(s.lower() for s in request.required_skills)
    preferred_skills_lower = set(s.lower() for s in request.preferred_skills)

    for sr in search_results:
        candidate = candidates.get(sr.candidate_id)
        if not candidate:
            continue

        # Calculate skill score
        candidate_skills = set(s.lower() for s in (candidate.top_skills or []))
        matched_required = required_skills_lower.intersection(candidate_skills)
        matched_preferred = preferred_skills_lower.intersection(candidate_skills)

        if required_skills_lower:
            skill_score = (
                len(matched_required) / len(required_skills_lower) * 70 +
                len(matched_preferred) / max(len(preferred_skills_lower), 1) * 30
            )
        else:
            skill_score = 50  # No specific requirements

        # Calculate experience score
        experience_score = 50  # Default
        if candidate.total_experience_years is not None:
            if request.min_experience_years:
                if candidate.total_experience_years >= request.min_experience_years:
                    experience_score = 100
                else:
                    experience_score = (
                        candidate.total_experience_years /
                        request.min_experience_years * 100
                    )
            else:
                experience_score = min(candidate.total_experience_years * 10, 100)

        # Semantic score from search
        semantic_score = sr.combined_score * 100  # Normalize

        # Weighted overall score
        overall_score = (
            skill_score * request.skill_weight +
            experience_score * request.experience_weight +
            semantic_score * request.semantic_weight
        )

        # Generate match explanation
        explanation_parts = []
        if matched_required:
            explanation_parts.append(
                f"Has required skills: {', '.join(matched_required)}"
            )
        if matched_preferred:
            explanation_parts.append(
                f"Has preferred skills: {', '.join(matched_preferred)}"
            )
        if candidate.total_experience_years:
            explanation_parts.append(
                f"{candidate.total_experience_years:.1f} years of experience"
            )

        matches.append(
            MatchResult(
                candidate_id=candidate.id,
                full_name=candidate.full_name,
                email=candidate.email,
                headline=candidate.headline,
                total_experience_years=candidate.total_experience_years,
                match_score=MatchScore(
                    overall_score=min(overall_score, 100),
                    skill_score=min(skill_score, 100),
                    experience_score=min(experience_score, 100),
                    semantic_score=min(semantic_score, 100),
                ),
                skills_matched=list(matched_required | matched_preferred),
                skills_missing=list(
                    required_skills_lower - candidate_skills
                ),
                experience_summary=(
                    f"{candidate.total_experience_years:.1f} years"
                    if candidate.total_experience_years else "N/A"
                ),
                match_explanation=" | ".join(explanation_parts) or "Semantic match to job description",
            )
        )

    return matches


@router.get("/skills/suggest")
async def suggest_skills(
    query: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Suggest skills based on query prefix.
    
    Used for autocomplete in search UI.
    """
    from sqlalchemy import select, func
    from app.models.candidate import Candidate

    # This is a simplified version - in production, you'd have a skills table
    result = await db.execute(
        select(
            func.unnest(Candidate.top_skills).label('skill')
        ).distinct().limit(limit)
    )

    skills = [row.skill for row in result if query.lower() in row.skill.lower()]

    return {"query": query, "suggestions": skills[:limit]}
