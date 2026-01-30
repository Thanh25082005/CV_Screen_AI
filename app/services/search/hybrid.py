"""
Hybrid Search Engine combining BM25, Vector, and RRF.

This is the main entry point for search operations in the system.
It orchestrates:
1. Query expansion (LLM-based)
2. BM25 keyword search
3. Vector semantic search
4. RRF fusion
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.services.search.bm25 import BM25Search
from app.services.search.vector import VectorSearch, VectorSearchResult
from app.services.search.query_expansion import QueryExpander, get_query_expander
from app.services.search.rrf import RRFMerger, RRFResult, get_rrf_merger
from app.schemas.search import SearchRequest, SearchResponse, SearchResult, ChunkMatch, SearchType
from app.config import get_settings
from app.services.utils.cache import get_cache

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    # RRF parameter
    rrf_k: int = 60

    # Search depth (how many results to fetch before fusion)
    bm25_fetch_k: int = 50
    vector_fetch_k: int = 50

    # Query expansion
    max_query_expansions: int = 5

    # Weights (for weighted RRF)
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


class HybridSearchEngine:
    """
    Main search engine combining keyword and semantic search.
    
    Flow:
    1. Receive search request
    2. Expand query using LLM
    3. Run BM25 search on expanded queries
    4. Run vector search on expanded queries
    5. Fuse results using RRF
    6. Return ranked candidates with matched chunks
    """

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        bm25_search: Optional[BM25Search] = None,
        vector_search: Optional[VectorSearch] = None,
        query_expander: Optional[QueryExpander] = None,
        rrf_merger: Optional[RRFMerger] = None,
    ):
        """
        Initialize hybrid search engine.
        
        All dependencies can be injected for testing.
        """
        self.config = config or HybridSearchConfig()
        self.bm25 = bm25_search or BM25Search()
        self.vector = vector_search or VectorSearch()
        self.expander = query_expander or get_query_expander()
        self.rrf = rrf_merger or get_rrf_merger(self.config.rrf_k)

    async def search(
        self,
        request: SearchRequest,
        session: AsyncSession,
    ) -> SearchResponse:
        """
        Execute hybrid search based on request parameters.
        
        Args:
            request: Search request with query and filters
            session: Database session
            
        Returns:
            SearchResponse with ranked results
        """
        start_time = time.time()

        # Step 0: Check Cache
        cache = get_cache()
        cache_key = request.model_dump()
        cached_results = await cache.get("search", cache_key)
        
        if cached_results:
            logger.info(f"Returning cached search results for query: {request.query}")
            return SearchResponse(
                results=[SearchResult(**r) for r in cached_results],
                total_results=len(cached_results),
                search_time_ms=(time.time() - start_time) * 1000,
                query=request.query,
                expanded_queries=[], # Cached results don't store expanded queries directly
                search_type=request.search_type,
            )

        # Step 1: Query expansion (if enabled)
        # Note: We now have separate semantic (request.query) and keyword (request.keyword_query) queries.
        
        # Expand Semantic Query (for filtering/vector)
        expanded_semantic_queries = [request.query]
        if request.expand_query:
             expanded_semantic_queries = self.expander.expand_query(
                request.query,
                max_expansions=self.config.max_query_expansions,
            )
        
        # Determine BM25 queries
        if request.keyword_query:
            # If explicit keyword query provided, use it (and maybe expand it too?)
            # For now, let's treat the explicit keyword query as the primary source for BM25
            bm25_queries = [request.keyword_query]
            logger.info(f"Using explicit keyword query for BM25: {bm25_queries}")
        else:
            # Fallback to expanding the main query
            bm25_queries = expanded_semantic_queries
            logger.info(f"Using expanded semantic queries for BM25: {bm25_queries}")

        # Step 1.5: Build pre-filters from request (NEW)
        filters = {}
        if request.location:
            filters["location"] = request.location
        if request.min_experience_years is not None:
            filters["min_experience_years"] = request.min_experience_years
        if request.required_skills:
            filters["required_skills"] = request.required_skills
        
        logger.info(f"Pre-filters applied: {filters}")

        # Step 2: Execute search based on type
        if request.search_type == SearchType.HYBRID:
            results = await self._hybrid_search(
                bm25_queries, expanded_semantic_queries, session, request.top_k, filters
            )
        elif request.search_type == SearchType.KEYWORD:
            results = await self._keyword_only_search(
                bm25_queries, session, request.top_k, filters
            )
        else:  # SEMANTIC
            results = await self._semantic_only_search(
                expanded_semantic_queries, session, request.top_k, filters
            )

        # Step 3: Apply post-filters (remaining filters not handled in pre-filter)
        filtered_results = self._apply_filters(results, request)

        # Step 3: Format and Store Cache
        search_time_ms = (time.time() - start_time) * 1000
        
        # Save to cache before returning
        await cache.set(
            "search", 
            cache_key, 
            [r.model_dump() for r in filtered_results[:request.top_k]], # Cache only the top_k results
            expire_seconds=300 # 5 minutes cache
        )

        return SearchResponse(
            query=request.query,
            expanded_queries=expanded_semantic_queries,  # Return semantic expansions for debug
            search_type=request.search_type,
            total_results=len(filtered_results),
            results=filtered_results[:request.top_k],
            search_time_ms=search_time_ms,
        )


    async def _hybrid_search(
        self,
        bm25_queries: List[str],
        vector_queries: List[str],
        session: AsyncSession,
        top_k: int,
        filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Execute hybrid BM25 + Vector search with RRF."""
        filters = filters or {}
        
        # Get BM25 results (BM25 doesn't support pre-filtering, will filter post-merge)
        bm25_results = self.bm25.search_with_expansion(
            bm25_queries, top_k=self.config.bm25_fetch_k
        )

        # Get vector results WITH PRE-FILTERING
        vector_raw_results = await self.vector.search_with_expanded_queries(
            vector_queries, session, top_k=self.config.vector_fetch_k, filters=filters
        )

        # Convert vector results to tuple format
        vector_results = [
            (r.chunk_id, r.candidate_id, r.similarity)
            for r in vector_raw_results
        ]

        # Fuse with RRF
        rrf_results = self.rrf.merge(bm25_results, vector_results)

        # Build SearchResult objects
        return await self._build_search_results(
            rrf_results, vector_raw_results, session
        )

    async def _keyword_only_search(
        self,
        queries: List[str],
        session: AsyncSession,
        top_k: int,
        filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Execute BM25-only search."""
        # Note: BM25 index doesn't support pre-filtering, filters applied post-search
        bm25_results = self.bm25.search_with_expansion(queries, top_k=top_k)

        # Build results with only keyword ranking
        results = []
        for rank, (doc_id, cand_id, score) in enumerate(bm25_results, 1):
            results.append(
                SearchResult(
                    candidate_id=cand_id,
                    full_name="",  # Will be populated later
                    combined_score=score,
                    keyword_rank=rank,
                    semantic_rank=None,
                    matched_chunks=[],
                )
            )

        return results

    async def _semantic_only_search(
        self,
        queries: List[str],
        session: AsyncSession,
        top_k: int,
        filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Execute vector-only search WITH PRE-FILTERING."""
        vector_results = await self.vector.search_with_expanded_queries(
            queries, session, top_k=top_k, filters=filters or {}
        )

        # Build results with only semantic ranking
        results = []
        for rank, vr in enumerate(vector_results, 1):
            results.append(
                SearchResult(
                    candidate_id=vr.candidate_id,
                    full_name=vr.full_name or "Unknown",
                    combined_score=vr.similarity,
                    keyword_rank=None,
                    semantic_rank=rank,
                    matched_chunks=[
                        ChunkMatch(
                            chunk_id=vr.chunk_id,
                            section=vr.section or "",
                            content=vr.content,
                            enriched_content=vr.enriched_content,
                            score=vr.similarity,
                            match_type="semantic",
                        )
                    ],
                )
            )

        return results

    async def _build_search_results(
        self,
        rrf_results: List[RRFResult],
        vector_raw: List[VectorSearchResult],
        session: AsyncSession,
    ) -> List[SearchResult]:
        """Build SearchResult objects from RRF fusion results."""
        
        # Create lookup for vector results
        vector_lookup = {vr.chunk_id: vr for vr in vector_raw}

        # Group results by candidate
        by_candidate: Dict[str, List[RRFResult]] = {}
        for rrf in rrf_results:
            if rrf.candidate_id not in by_candidate:
                by_candidate[rrf.candidate_id] = []
            by_candidate[rrf.candidate_id].append(rrf)

        # Build results for each candidate
        results = []
        for cand_id, chunks in by_candidate.items():
            # Aggregate score
            combined_score = sum(c.combined_score for c in chunks)

            # Get best ranks
            keyword_ranks = [c.keyword_rank for c in chunks if c.keyword_rank]
            semantic_ranks = [c.semantic_rank for c in chunks if c.semantic_rank]

            # Build chunk matches
            matched_chunks = []
            for rrf in chunks:
                vr = vector_lookup.get(rrf.doc_id)
                
                # Determine match type
                match_type = "hybrid"
                if rrf.keyword_rank and not rrf.semantic_rank:
                    match_type = "keyword"
                elif rrf.semantic_rank and not rrf.keyword_rank:
                    match_type = "semantic"

                matched_chunks.append(
                    ChunkMatch(
                        chunk_id=rrf.doc_id,
                        section=vr.section if vr else "",
                        subsection=vr.metadata.get("subsection") if vr and vr.metadata else None,
                        content=vr.content if vr else "",
                        enriched_content=vr.enriched_content if vr else None,
                        score=rrf.combined_score,
                        match_type=match_type,
                    )
                )

            # Find full_name and top_skills from vector results if available
            candidate_full_name = "Unknown"
            candidate_top_skills = []
            
            for rrf in chunks:
                vr = vector_lookup.get(rrf.doc_id)
                if vr:
                    if vr.full_name and candidate_full_name == "Unknown":
                        candidate_full_name = vr.full_name
                    if vr.top_skills and not candidate_top_skills:
                        candidate_top_skills = vr.top_skills
                
                if candidate_full_name != "Unknown" and candidate_top_skills:
                    break

            results.append(
                SearchResult(
                    candidate_id=cand_id,
                    full_name=candidate_full_name,
                    top_skills=candidate_top_skills,
                    combined_score=combined_score,
                    keyword_rank=min(keyword_ranks) if keyword_ranks else None,
                    semantic_rank=min(semantic_ranks) if semantic_ranks else None,
                    matched_chunks=matched_chunks,
                )
            )

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results

    def _apply_filters(
        self,
        results: List[SearchResult],
        request: SearchRequest,
    ) -> List[SearchResult]:
        """Apply post-search filters."""
        filtered = results

        # Filter by section if specified
        if request.include_sections:
            section_set = set(s.lower() for s in request.include_sections)
            for result in filtered:
                result.matched_chunks = [
                    c for c in result.matched_chunks
                    if c.section.lower() in section_set
                ]
            # Remove results with no matching chunks
            filtered = [r for r in filtered if r.matched_chunks]

        # Additional filters would go here (experience, skills, etc.)
        # These require fetching candidate data from DB

        return filtered

    def update_bm25_index(
        self,
        doc_ids: List[str],
        candidate_ids: List[str],
        contents: List[str],
    ) -> None:
        """
        Update the BM25 index with new documents.
        
        Should be called after adding new candidates.
        """
        self.bm25.add_documents(doc_ids, candidate_ids, contents)
        logger.info(f"Updated BM25 index with {len(doc_ids)} documents")

    def refresh_bm25_index_sync(self, session) -> int:
        """
        Refresh BM25 index from database (synchronous version for startup).
        
        Args:
            session: SQLAlchemy session (sync)
            
        Returns:
            Number of documents indexed
        """
        return self.bm25.load_from_database(session)

    def add_candidate_chunks(
        self,
        chunk_ids: List[str],
        candidate_id: str,
        contents: List[str],
    ) -> None:
        """
        Add chunks for a new candidate to BM25 index incrementally.
        
        Note: BM25 requires re-indexing for proper IDF calculation,
        so this adds to existing documents and re-indexes all.
        
        Args:
            chunk_ids: List of chunk IDs
            candidate_id: The candidate ID these chunks belong to
            contents: List of enriched content for each chunk
        """
        from app.services.search.bm25 import BM25Document
        
        # Get existing documents
        current_docs = list(self.bm25.documents)
        
        # Add new documents
        for chunk_id, content in zip(chunk_ids, contents):
            current_docs.append(BM25Document(
                id=chunk_id,
                candidate_id=candidate_id,
                content=content,
            ))
        
        # Re-index all documents
        self.bm25.index_documents(current_docs)
        logger.info(f"Added {len(chunk_ids)} chunks for candidate {candidate_id} to BM25 index")


# Singleton instance
_engine: Optional[HybridSearchEngine] = None


def get_search_engine() -> HybridSearchEngine:
    """Get or create the hybrid search engine singleton."""
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine

