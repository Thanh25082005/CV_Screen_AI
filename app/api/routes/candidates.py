"""
Candidate Management API Routes.

Endpoints for:
- Listing candidates
- Getting candidate details
- Finding similar candidates
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.api.deps import get_async_db
from app.models.candidate import Candidate
from app.schemas.validation import CandidateResponse, CandidateListResponse
from app.services.search.vector import VectorSearch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.get("", response_model=CandidateListResponse)
async def list_candidates(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    skills: Optional[str] = Query(None, description="Filter by skills (comma-separated)"),
    min_experience: Optional[float] = Query(None, ge=0, description="Minimum experience years"),
    db: AsyncSession = Depends(get_async_db),
):
    """
    List all candidates with pagination and filtering.
    """
    # Base query
    query = select(Candidate)

    # Apply filters
    if min_experience is not None:
        query = query.where(Candidate.total_experience_years >= min_experience)

    # Apply sorting
    if sort_order == "desc":
        query = query.order_by(desc(getattr(Candidate, sort_by, Candidate.created_at)))
    else:
        query = query.order_by(getattr(Candidate, sort_by, Candidate.created_at))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # Execute
    result = await db.execute(query)
    candidates = result.scalars().all()

    return CandidateListResponse(
        total=total,
        page=page,
        page_size=page_size,
        candidates=[
            CandidateResponse(
                id=c.id,
                full_name=c.full_name,
                email=c.email,
                phone=c.phone,
                headline=c.headline,
                summary=c.summary,
                total_experience_years=c.total_experience_years,
                top_skills=c.top_skills or [],
                validation_warnings=c.validation_warnings or [],
                created_at=c.created_at,
            )
            for c in candidates
        ],
    )


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(
    candidate_id: str,
    include_resume_data: bool = Query(False, description="Include full resume JSON"),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get detailed information about a specific candidate.
    """
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    return CandidateResponse(
        id=candidate.id,
        full_name=candidate.full_name,
        email=candidate.email,
        phone=candidate.phone,
        headline=candidate.headline,
        summary=candidate.summary,
        total_experience_years=candidate.total_experience_years,
        top_skills=candidate.top_skills or [],
        validation_warnings=candidate.validation_warnings or [],
        created_at=candidate.created_at,
        resume_data=candidate.raw_resume if include_resume_data else None,
    )


@router.get("/{candidate_id}/similar")
async def get_similar_candidates(
    candidate_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of similar candidates"),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Find candidates similar to the specified candidate.
    
    Uses vector similarity on profile summaries.
    """
    # Verify candidate exists
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    # Find similar candidates
    vector_search = VectorSearch()
    similar = await vector_search.find_similar_candidates(
        candidate_id, db, top_k=top_k
    )

    # Enrich with full candidate data
    similar_ids = [s[0] for s in similar]
    if similar_ids:
        result = await db.execute(
            select(Candidate).where(Candidate.id.in_(similar_ids))
        )
        candidates_map = {c.id: c for c in result.scalars().all()}
    else:
        candidates_map = {}

    return {
        "reference_candidate": {
            "id": candidate.id,
            "full_name": candidate.full_name,
        },
        "similar_candidates": [
            {
                "id": cand_id,
                "full_name": candidates_map[cand_id].full_name if cand_id in candidates_map else None,
                "headline": candidates_map[cand_id].headline if cand_id in candidates_map else None,
                "similarity_score": score,
            }
            for cand_id, _, score in similar
            if cand_id in candidates_map
        ],
    }


@router.get("/{candidate_id}/experience")
async def get_candidate_experience(
    candidate_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get detailed work experience for a candidate.
    
    Includes merged experience calculation.
    """
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    resume = candidate.raw_resume or {}
    work_experience = resume.get("work_experience", [])

    return {
        "candidate_id": candidate_id,
        "total_experience_years": candidate.total_experience_years,
        "positions_count": len(work_experience),
        "work_experience": work_experience,
    }


@router.delete("/{candidate_id}")
async def delete_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Delete a candidate and all associated data.
    
    This removes:
    - Candidate record
    - All chunks (cascade)
    - Associated embeddings
    """
    from sqlalchemy import delete

    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    await db.execute(
        delete(Candidate).where(Candidate.id == candidate_id)
    )
    await db.commit()

    logger.info(f"Deleted candidate: {candidate_id}")

    return {"message": f"Candidate {candidate_id} deleted successfully"}


@router.delete("/all/clear", status_code=status.HTTP_200_OK)
async def delete_all_candidates(
    db: AsyncSession = Depends(get_async_db),
):
    """
    Delete ALL candidates and associated data from the database.
    
    WARNING: This action is irreversible.
    """
    from sqlalchemy import delete
    
    try:
        # Get count before deletion
        count_result = await db.execute(select(func.count(Candidate.id)))
        total_before = count_result.scalar() or 0
        
        # Delete all candidates (cascades to chunks)
        await db.execute(delete(Candidate))
        await db.commit()
        
        logger.warning(f"DATABASE CLEANUP: Deleted all {total_before} candidates.")
        
        return {
            "message": "All candidates and associated data have been deleted successfully.",
            "deleted_count": total_before
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear database: {str(e)}"
        )


@router.get("/stats/overview")
async def get_candidates_stats(
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get overview statistics for all candidates.
    """
    # Total candidates
    total_result = await db.execute(select(func.count(Candidate.id)))
    total = total_result.scalar() or 0

    # Average experience
    avg_exp_result = await db.execute(
        select(func.avg(Candidate.total_experience_years))
    )
    avg_experience = avg_exp_result.scalar() or 0

    # Candidates with validation warnings
    warnings_result = await db.execute(
        select(func.count()).where(
            func.jsonb_array_length(Candidate.validation_warnings) > 0
        )
    )
    with_warnings = warnings_result.scalar() or 0

    return {
        "total_candidates": total,
        "average_experience_years": round(avg_experience, 1),
        "candidates_with_warnings": with_warnings,
    }
