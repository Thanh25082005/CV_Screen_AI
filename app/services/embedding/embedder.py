"""
Embedding Service using Sentence Transformers.

Provides vector embeddings for CV chunks and search queries using:
- BAAI/bge-m3: Multilingual model with excellent Vietnamese/English support
- keepitreal/vietnamese-sbert: Vietnamese-focused alternative

The service handles:
1. Text preprocessing before embedding
2. Batch embedding for efficiency
3. Caching of the model for repeated use
"""

import logging
from typing import List, Optional, Union
import numpy as np

from app.config import get_settings
from app.services.ingestion.preprocessor import get_preprocessor

settings = get_settings()
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using Sentence Transformers.
    
    Features:
    - Lazy model loading (only loads when first needed)
    - Vietnamese text preprocessing before embedding
    - Batch embedding support
    - Configurable model selection
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        preprocess_vietnamese: bool = True,
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Sentence-transformers model name
            cache_dir: Directory to cache downloaded models
            preprocess_vietnamese: Apply Vietnamese word segmentation
        """
        self.model_name = model_name or settings.embedding_model
        self.cache_dir = cache_dir or settings.model_cache_dir
        self.preprocess_vietnamese = preprocess_vietnamese

        self._model = None
        self._initialized = False
        self._preprocessor = get_preprocessor() if preprocess_vietnamese else None

    def _lazy_init(self):
        """Lazy initialization of the embedding model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Smart device selection
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Attempting to use device: {device}")
            
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=device
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("CUDA Out of Memory! Falling back to CPU for embeddings.")
                    torch.cuda.empty_cache()
                    device = "cpu"
                    self._model = SentenceTransformer(
                        self.model_name,
                        cache_folder=self.cache_dir,
                        device="cpu"
                    )
                else:
                    raise e
                    
            self._initialized = True
            logger.info(
                f"Model loaded on {device}. Dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        self._lazy_init()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as list of floats
        """
        self._lazy_init()

        # Preprocess Vietnamese text
        if self.preprocess_vietnamese and self._preprocessor:
            text = self._preprocessor.preprocess_for_embedding(text)

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            List of embeddings
        """
        self._lazy_init()

        if not texts:
            return []

        # Preprocess all texts
        if self.preprocess_vietnamese and self._preprocessor:
            texts = [
                self._preprocessor.preprocess_for_embedding(t) for t in texts
            ]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # Normalize for cosine similarity
        query_norm = query / np.linalg.norm(query)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Calculate all similarities at once
        similarities = np.dot(candidates_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        
        For some models, query embedding uses a different prefix.
        This method handles that automatically.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding
        """
        self._lazy_init()

        # Some models like BGE use special prefixes for queries
        if "bge" in self.model_name.lower():
            # BGE models recommend adding "query: " prefix
            query = f"query: {query}"

        return self.embed(query)

    def embed_document(self, document: str) -> List[float]:
        """
        Embed a document (CV chunk).
        
        For some models, document embedding uses a different prefix.
        
        Args:
            document: Document text
            
        Returns:
            Document embedding
        """
        self._lazy_init()

        # BGE models can use "passage: " prefix for documents
        # but it's optional, so we just use regular embedding
        return self.embed(document)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
