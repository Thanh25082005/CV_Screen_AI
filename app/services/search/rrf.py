"""
Reciprocal Rank Fusion (RRF) Implementation.

RRF is a technique for combining results from multiple search methods
(e.g., BM25 keyword search + vector semantic search).

Formula: Score = Σ 1/(k + rank_i)

Where:
- k is a constant (typically 60)
- rank_i is the rank of the document in search method i

Key benefits:
- Doesn't require score normalization across different methods
- Prioritizes documents that rank highly in multiple methods
- Simple yet effective fusion strategy
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RRFResult:
    """Result from RRF fusion."""

    doc_id: str
    candidate_id: str
    combined_score: float
    keyword_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    semantic_score: Optional[float] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RRFMerger:
    """
    Reciprocal Rank Fusion for combining keyword and semantic search results.
    
    The RRF formula assigns higher scores to documents that appear
    near the top of multiple result lists.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF merger.
        
        Args:
            k: RRF constant. Larger values reduce the penalty for lower ranks.
               Common values: 60 (standard), 20 (more emphasis on top ranks)
        """
        self.k = k

    def merge(
        self,
        bm25_results: List[Tuple[str, str, float]],  # (doc_id, cand_id, score)
        vector_results: List[Tuple[str, str, float]],
    ) -> List[RRFResult]:
        """
        Merge BM25 and vector search results using RRF.
        
        Args:
            bm25_results: Results from BM25 search (doc_id, candidate_id, score)
            vector_results: Results from vector search (doc_id, candidate_id, score)
            
        Returns:
            Merged results sorted by RRF score
        """
        # Track scores and metadata for each document
        doc_data: Dict[str, Dict[str, Any]] = {}

        # Process BM25 results
        for rank, (doc_id, cand_id, score) in enumerate(bm25_results, 1):
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "candidate_id": cand_id,
                    "rrf_score": 0.0,
                }
            doc_data[doc_id]["keyword_rank"] = rank
            doc_data[doc_id]["keyword_score"] = score
            doc_data[doc_id]["rrf_score"] += 1.0 / (self.k + rank)

        # Process vector results
        for rank, (doc_id, cand_id, score) in enumerate(vector_results, 1):
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "candidate_id": cand_id,
                    "rrf_score": 0.0,
                }
            doc_data[doc_id]["semantic_rank"] = rank
            doc_data[doc_id]["semantic_score"] = score
            doc_data[doc_id]["rrf_score"] += 1.0 / (self.k + rank)

        # Build result list
        results = []
        for doc_id, data in doc_data.items():
            results.append(
                RRFResult(
                    doc_id=doc_id,
                    candidate_id=data["candidate_id"],
                    combined_score=data["rrf_score"],
                    keyword_rank=data.get("keyword_rank"),
                    semantic_rank=data.get("semantic_rank"),
                    keyword_score=data.get("keyword_score"),
                    semantic_score=data.get("semantic_score"),
                )
            )

        # Sort by combined RRF score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        logger.info(
            f"RRF merged {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"results into {len(results)} unique documents"
        )

        return results

    def merge_with_weights(
        self,
        result_lists: List[Tuple[List[Tuple[str, str, float]], float]],
    ) -> List[RRFResult]:
        """
        Merge multiple result lists with custom weights.
        
        Args:
            result_lists: List of (results, weight) tuples.
                Each results list contains (doc_id, candidate_id, score) tuples.
            
        Returns:
            Merged results sorted by weighted RRF score
        """
        doc_data: Dict[str, Dict[str, Any]] = {}

        for results, weight in result_lists:
            for rank, (doc_id, cand_id, score) in enumerate(results, 1):
                if doc_id not in doc_data:
                    doc_data[doc_id] = {
                        "candidate_id": cand_id,
                        "rrf_score": 0.0,
                    }
                # Apply weight to RRF score
                doc_data[doc_id]["rrf_score"] += weight / (self.k + rank)

        results = [
            RRFResult(
                doc_id=doc_id,
                candidate_id=data["candidate_id"],
                combined_score=data["rrf_score"],
            )
            for doc_id, data in doc_data.items()
        ]

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def merge_candidate_level(
        self,
        bm25_results: List[Tuple[str, str, float]],
        vector_results: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Merge results at candidate level (aggregate by candidate_id).
        
        For each candidate, sum the RRF scores of all their chunks.
        
        Args:
            bm25_results: BM25 results (doc_id, candidate_id, score)
            vector_results: Vector results (doc_id, candidate_id, score)
            
        Returns:
            List of (candidate_id, total_score) sorted by score
        """
        # First do document-level RRF
        doc_results = self.merge(bm25_results, vector_results)

        # Aggregate by candidate
        candidate_scores: Dict[str, float] = defaultdict(float)

        for result in doc_results:
            candidate_scores[result.candidate_id] += result.combined_score

        # Sort by total score
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_candidates

    @staticmethod
    def explain_ranking(result: RRFResult) -> str:
        """
        Generate explanation for why a result ranked where it did.
        
        Args:
            result: RRF result to explain
            
        Returns:
            Human-readable explanation
        """
        parts = []

        if result.keyword_rank is not None:
            parts.append(f"Keyword rank #{result.keyword_rank}")
        else:
            parts.append("No keyword match")

        if result.semantic_rank is not None:
            parts.append(f"Semantic rank #{result.semantic_rank}")
        else:
            parts.append("No semantic match")

        parts.append(f"Combined RRF score: {result.combined_score:.4f}")

        return " | ".join(parts)


# Module-level instance
_merger: Optional[RRFMerger] = None


def get_rrf_merger(k: int = 60) -> RRFMerger:
    """Get or create an RRF merger instance."""
    global _merger
    if _merger is None or _merger.k != k:
        _merger = RRFMerger(k=k)
    return _merger
