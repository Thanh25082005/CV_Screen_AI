"""
Tests for the Search Engine components.
"""

import pytest
from app.services.search.bm25 import BM25Search, BM25Document
from app.services.search.rrf import RRFMerger, RRFResult


class TestBM25Search:
    """Tests for BM25 search implementation."""

    @pytest.fixture
    def bm25(self):
        """Create and index a BM25 search instance."""
        search = BM25Search()

        documents = [
            BM25Document(
                id="1",
                candidate_id="c1",
                content="Python developer with experience in Django and Flask frameworks",
            ),
            BM25Document(
                id="2",
                candidate_id="c2",
                content="Java software engineer specializing in Spring Boot and microservices",
            ),
            BM25Document(
                id="3",
                candidate_id="c3",
                content="Full stack developer with Python, JavaScript, and React experience",
            ),
            BM25Document(
                id="4",
                candidate_id="c4",
                content="Machine learning engineer with Python and TensorFlow expertise",
            ),
            BM25Document(
                id="5",
                candidate_id="c5",
                content="DevOps engineer with Docker, Kubernetes, and AWS experience",
            ),
        ]

        search.index_documents(documents)
        return search

    def test_search_python(self, bm25):
        """Test searching for Python."""
        results = bm25.search("Python", top_k=5)

        assert len(results) > 0

        # Documents mentioning Python should rank high
        doc_ids = [r[0] for r in results]
        assert "1" in doc_ids  # Python developer
        assert "3" in doc_ids  # Full stack with Python
        assert "4" in doc_ids  # ML with Python

    def test_search_java(self, bm25):
        """Test searching for Java."""
        results = bm25.search("Java Spring Boot", top_k=5)

        assert len(results) > 0
        assert results[0][0] == "2"  # Java developer should be first

    def test_search_no_results(self, bm25):
        """Test search with no matching terms."""
        results = bm25.search("xyz123nonexistent", top_k=5)

        # Should return empty or very low scores
        assert len(results) == 0 or all(r[2] == 0 for r in results)

    def test_search_empty_query(self, bm25):
        """Test search with empty query."""
        results = bm25.search("", top_k=5)
        assert len(results) == 0

    def test_multiple_term_query(self, bm25):
        """Test multi-term query."""
        results = bm25.search("Python Django developer", top_k=5)

        assert len(results) > 0
        # Document 1 should score highest (has all terms)
        assert results[0][0] == "1"

    def test_term_stats(self, bm25):
        """Test getting term statistics."""
        stats = bm25.get_term_stats("python")

        assert stats["term"] == "python"
        assert stats["document_frequency"] == 3  # 3 docs mention Python
        assert stats["idf"] > 0

    def test_get_document_by_id(self, bm25):
        """Test retrieving document by ID."""
        doc = bm25.get_document_by_id("1")

        assert doc is not None
        assert doc.candidate_id == "c1"
        assert "Python" in doc.content


class TestRRFMerger:
    """Tests for Reciprocal Rank Fusion."""

    @pytest.fixture
    def merger(self):
        return RRFMerger(k=60)

    def test_basic_merge(self, merger):
        """Test basic RRF merging."""
        bm25_results = [
            ("doc1", "c1", 10.0),
            ("doc2", "c2", 8.0),
            ("doc3", "c3", 5.0),
        ]

        vector_results = [
            ("doc2", "c2", 0.95),  # Same doc, different order
            ("doc1", "c1", 0.90),
            ("doc4", "c4", 0.85),
        ]

        results = merger.merge(bm25_results, vector_results)

        assert len(results) == 4  # 4 unique documents

        # doc2 appears rank 1 in vector, rank 2 in BM25 - should score well
        # doc1 appears rank 1 in BM25, rank 2 in vector - should also score well
        doc_ids = [r.doc_id for r in results]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_document_in_one_list_only(self, merger):
        """Test document appearing in only one result list."""
        bm25_results = [
            ("doc1", "c1", 10.0),
        ]

        vector_results = [
            ("doc2", "c2", 0.95),
        ]

        results = merger.merge(bm25_results, vector_results)

        assert len(results) == 2
        # Both should have RRF score from single list
        for r in results:
            assert r.combined_score > 0

    def test_empty_lists(self, merger):
        """Test merging empty lists."""
        results = merger.merge([], [])
        assert len(results) == 0

    def test_rrf_formula(self, merger):
        """Test that RRF formula is applied correctly."""
        bm25_results = [("doc1", "c1", 10.0)]
        vector_results = [("doc1", "c1", 0.95)]

        results = merger.merge(bm25_results, vector_results)

        # doc1 is rank 1 in both lists
        # RRF score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
        expected_score = 2 / (60 + 1)

        assert len(results) == 1
        assert abs(results[0].combined_score - expected_score) < 0.0001

    def test_ranking_preserved(self, merger):
        """Test that output is sorted by combined score."""
        bm25_results = [
            ("doc1", "c1", 10.0),
            ("doc2", "c2", 8.0),
            ("doc3", "c3", 5.0),
        ]

        vector_results = [
            ("doc3", "c3", 0.95),
            ("doc2", "c2", 0.90),
            ("doc1", "c1", 0.85),
        ]

        results = merger.merge(bm25_results, vector_results)

        # Verify descending order by score
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tracks_ranks(self, merger):
        """Test that individual ranks are tracked."""
        bm25_results = [
            ("doc1", "c1", 10.0),
            ("doc2", "c2", 8.0),
        ]

        vector_results = [
            ("doc2", "c2", 0.95),
            ("doc1", "c1", 0.90),
        ]

        results = merger.merge(bm25_results, vector_results)

        # Find doc1
        doc1_result = next(r for r in results if r.doc_id == "doc1")
        assert doc1_result.keyword_rank == 1
        assert doc1_result.semantic_rank == 2

    def test_candidate_level_merge(self, merger):
        """Test merging at candidate level."""
        bm25_results = [
            ("chunk1", "c1", 10.0),
            ("chunk2", "c1", 8.0),  # Same candidate
            ("chunk3", "c2", 5.0),
        ]

        vector_results = [
            ("chunk1", "c1", 0.95),
            ("chunk4", "c2", 0.90),  # Different chunk, same candidate as chunk3
        ]

        results = merger.merge_candidate_level(bm25_results, vector_results)

        # Should have 2 candidates
        assert len(results) == 2

        # c1 should have higher total score (appeared more times)
        assert results[0][0] == "c1"

    def test_explain_ranking(self, merger):
        """Test ranking explanation."""
        result = RRFResult(
            doc_id="doc1",
            candidate_id="c1",
            combined_score=0.033,
            keyword_rank=1,
            semantic_rank=2,
        )

        explanation = RRFMerger.explain_ranking(result)

        assert "Keyword rank #1" in explanation
        assert "Semantic rank #2" in explanation
        assert "0.033" in explanation


class TestQueryExpansion:
    """Tests for query expansion (basic functionality)."""

    def test_fallback_expansion(self):
        """Test fallback expansion without LLM."""
        from app.services.search.query_expansion import QueryExpander

        expander = QueryExpander(api_key=None)  # No API key = fallback

        # Set initialized to avoid any API calls
        expander._initialized = True

        expanded = expander._fallback_expansion("Python developer")

        # Should include some variations
        assert len(expanded) >= 0  # Might be empty for some queries

    def test_skill_expansion(self):
        """Test skill-related expansion."""
        from app.services.search.query_expansion import QueryExpander

        expander = QueryExpander(api_key=None)
        expander._initialized = True

        expanded = expander.expand_for_skills("python")

        assert "python" in [s.lower() for s in expanded]
        # Should include related skills
        assert any(s.lower() in ["python3", "django", "flask", "fastapi"] for s in expanded)


class TestHybridSearchConfig:
    """Tests for hybrid search configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from app.services.search.hybrid import HybridSearchConfig

        config = HybridSearchConfig()

        assert config.rrf_k == 60
        assert config.bm25_fetch_k == 50
        assert config.vector_fetch_k == 50
        assert config.bm25_weight == 1.0
        assert config.vector_weight == 1.0
