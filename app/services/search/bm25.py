"""
BM25 Keyword Search Implementation.

BM25 (Best Matching 25) is a ranking function for keyword-based search
that considers:
- Term frequency (how often a term appears in a document)
- Inverse document frequency (how rare a term is across all documents)
- Document length normalization

This is used for exact keyword matching of hard skills like:
- "Python", "ReactJS", "PMP", "AWS"

Works alongside vector search for hybrid retrieval.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """Document in the BM25 index."""

    id: str
    candidate_id: str
    content: str
    tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Search:
    """
    BM25 search implementation for keyword matching.
    
    Uses the Okapi BM25 algorithm for ranking documents by keyword relevance.
    
    Key parameters:
    - k1: Term saturation parameter (typically 1.2 to 2.0)
    - b: Length normalization (0.75 is standard)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 search.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            epsilon: Small constant for IDF smoothing
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.documents: List[BM25Document] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for keyword search.
        
        For Vietnamese, words with underscores (from segmentation)
        are kept as single tokens.
        """
        # Convert to lowercase
        text = text.lower()

        # Replace punctuation with spaces (except underscores)
        import re
        text = re.sub(r"[^\w\s]", " ", text)

        # Split on whitespace
        tokens = text.split()

        # Filter very short tokens
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def index_documents(self, documents: List[BM25Document]) -> None:
        """
        Build the BM25 index from documents.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents

        # Tokenize all documents
        for doc in self.documents:
            doc.tokens = self._tokenize(doc.content)

        # Calculate document lengths
        self.doc_lengths = [len(doc.tokens) for doc in self.documents]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0
        )

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.documents:
            unique_tokens = set(doc.tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate IDF for all terms
        n_docs = len(self.documents)
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # IDF with smoothing
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = max(idf, self.epsilon)

        self.indexed = True
        logger.info(
            f"BM25 indexed {n_docs} documents, "
            f"{len(self.doc_freqs)} unique terms, "
            f"avgdl={self.avgdl:.1f}"
        )

    def add_documents(
        self,
        doc_ids: List[str],
        candidate_ids: List[str],
        contents: List[str],
        metadata_list: Optional[List[Dict]] = None,
    ) -> None:
        """
        Convenience method to add documents from raw data.
        
        Args:
            doc_ids: Document/chunk IDs
            candidate_ids: Associated candidate IDs
            contents: Text contents
            metadata_list: Optional metadata for each document
        """
        if metadata_list is None:
            metadata_list = [{} for _ in doc_ids]

        documents = [
            BM25Document(
                id=doc_id,
                candidate_id=cand_id,
                content=content,
                metadata=metadata,
            )
            for doc_id, cand_id, content, metadata in zip(
                doc_ids, candidate_ids, contents, metadata_list
            )
        ]

        self.index_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, candidate_id, score) tuples, sorted by score
        """
        if not self.indexed:
            logger.warning("BM25 index not built. Call index_documents first.")
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = []
        for i, doc in enumerate(self.documents):
            score = self._score_document(doc, query_tokens, i)
            scores.append((doc.id, doc.candidate_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)

        return scores[:top_k]

    def _score_document(
        self,
        doc: BM25Document,
        query_tokens: List[str],
        doc_idx: int,
    ) -> float:
        """Calculate BM25 score for a document against query tokens."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        tf = {}
        for token in doc.tokens:
            tf[token] = tf.get(token, 0) + 1

        for token in query_tokens:
            if token not in self.idf:
                continue

            # Term frequency in this document
            freq = tf.get(token, 0)
            if freq == 0:
                continue

            # BM25 score formula
            idf = self.idf[token]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avgdl)
            )

            score += idf * (numerator / denominator)

        return score

    def search_with_expansion(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Search with multiple query variations (from query expansion).
        
        Combines scores from all query variations using max.
        
        Args:
            queries: List of query variations
            top_k: Number of results to return
            
        Returns:
            Combined results sorted by best score
        """
        combined_scores: Dict[str, Tuple[str, float]] = {}

        for query in queries:
            results = self.search(query, top_k=top_k * 2)

            for doc_id, cand_id, score in results:
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = (cand_id, score)
                else:
                    # Keep max score
                    if score > combined_scores[doc_id][1]:
                        combined_scores[doc_id] = (cand_id, score)

        # Convert to list and sort
        results = [
            (doc_id, cand_id, score)
            for doc_id, (cand_id, score) in combined_scores.items()
        ]
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:top_k]

    def get_document_by_id(self, doc_id: str) -> Optional[BM25Document]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_term_stats(self, term: str) -> Dict[str, Any]:
        """Get statistics for a term in the index."""
        term = term.lower()
        return {
            "term": term,
            "document_frequency": self.doc_freqs.get(term, 0),
            "idf": self.idf.get(term, 0),
            "total_documents": len(self.documents),
        }

    def load_from_database(self, session: Session) -> int:
        """
        Load all chunks from database and build BM25 index.
        
        Args:
            session: SQLAlchemy database session
            
        Returns:
            Number of documents indexed
        """
        from app.models.candidate import Chunk
        
        # Query all chunks with enriched content
        chunks = session.query(Chunk).all()
        
        if not chunks:
            logger.info("No chunks found in database for BM25 indexing")
            return 0
        
        documents = [
            BM25Document(
                id=chunk.id,
                candidate_id=chunk.candidate_id,
                content=chunk.enriched_content or chunk.content,
            )
            for chunk in chunks
        ]
        
        self.index_documents(documents)
        logger.info(f"BM25 index loaded with {len(documents)} chunks from database")
        return len(documents)
