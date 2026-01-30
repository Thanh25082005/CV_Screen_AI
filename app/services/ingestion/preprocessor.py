"""
Vietnamese NLP Preprocessing using Underthesea.

Key functionality:
- Word segmentation for Vietnamese text
- This helps embedding models understand compound words like:
  "trí tuệ nhân tạo" (artificial intelligence) as a single concept
  instead of 4 separate words: "trí", "tuệ", "nhân", "tạo"
"""

import logging
import re
from typing import Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)


class VietnamesePreprocessor:
    """
    Vietnamese text preprocessor using Underthesea.
    
    Provides word segmentation which is critical for Vietnamese NLP because:
    - Vietnamese words can be multi-syllable (each syllable is a separate token)
    - Without segmentation, embeddings treat each syllable independently
    - Proper segmentation groups syllables into meaningful words
    
    Example:
        "kỹ sư phần mềm" → "kỹ_sư phần_mềm" (software engineer)
        This helps the embedding model understand these are compound words.
    """

    def __init__(self):
        self._underthesea = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of Underthesea."""
        if self._initialized:
            return

        try:
            import underthesea
            self._underthesea = underthesea
            self._initialized = True
            logger.info("Underthesea Vietnamese NLP initialized")
        except ImportError:
            logger.warning(
                "Underthesea not installed. Vietnamese word segmentation disabled. "
                "Install with: pip install underthesea"
            )

    def segment_words(self, text: str, join_char: str = "_") -> str:
        """
        Segment Vietnamese text into words.
        
        Compound words are joined with the specified character.
        
        Args:
            text: Vietnamese text to segment
            join_char: Character to join syllables of compound words
            
        Returns:
            Segmented text with compound words joined
            
        Example:
            >>> preprocessor.segment_words("Tôi là kỹ sư phần mềm")
            "Tôi là kỹ_sư phần_mềm"
        """
        self._lazy_init()

        if not self._underthesea:
            return text

        try:
            # word_tokenize returns segmented text with spaces
            # format="text" returns string with underscores for compound words
            segmented = self._underthesea.word_tokenize(text, format="text")
            return segmented
        except Exception as e:
            logger.warning(f"Word segmentation failed: {e}")
            return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text into a list of words.
        
        Args:
            text: Vietnamese text to tokenize
            
        Returns:
            List of tokens (compound words included)
        """
        self._lazy_init()

        if not self._underthesea:
            return text.split()

        try:
            return self._underthesea.word_tokenize(text)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return text.split()

    def pos_tag(self, text: str) -> List[tuple]:
        """
        Part-of-speech tagging for Vietnamese text.
        
        Useful for identifying named entities like company names, job titles.
        
        Args:
            text: Vietnamese text
            
        Returns:
            List of (word, pos_tag) tuples
        """
        self._lazy_init()

        if not self._underthesea:
            return [(word, "UNKNOWN") for word in text.split()]

        try:
            return self._underthesea.pos_tag(text)
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}")
            return [(word, "UNKNOWN") for word in text.split()]

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Vietnamese or English.
        
        Uses heuristics based on Vietnamese-specific characters.
        
        Returns:
            "vi" for Vietnamese, "en" for English, "mixed" for mixed content
        """
        # Vietnamese-specific characters
        vietnamese_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                               "ìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                               "ùúủũụưừứửữựỳýỷỹỵđ"
                               "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆ"
                               "ÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
                               "ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")

        text_chars = set(text)
        vn_char_count = len(text_chars.intersection(vietnamese_chars))

        if vn_char_count > 5:
            return "vi"
        elif vn_char_count > 0:
            return "mixed"
        else:
            return "en"

    def preprocess_for_embedding(self, text: str) -> str:
        """
        Full preprocessing pipeline for Vietnamese text before embedding.
        
        Steps:
        1. Detect language
        2. Apply word segmentation if Vietnamese
        3. Normalize whitespace
        4. Clean special characters
        
        Args:
            text: Raw text from CV
            
        Returns:
            Preprocessed text ready for embedding
        """
        if not text or not text.strip():
            return ""

        # Detect language
        lang = self.detect_language(text)

        # Apply segmentation for Vietnamese content
        if lang in ("vi", "mixed"):
            text = self.segment_words(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Clean but preserve meaningful punctuation
        text = self._clean_text(text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r"[^\S\n]+", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving Vietnamese characters and meaningful punctuation."""
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Keep Vietnamese chars, alphanumeric, common punctuation, and whitespace
        # This regex allows Vietnamese diacritics
        text = re.sub(
            r"[^\w\s\-_.,;:!?@#$%&*()\[\]{}\"\'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
            r"ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
            r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
            r"ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]+",
            " ",
            text,
        )

        return text


# Singleton instance
_preprocessor: Optional[VietnamesePreprocessor] = None


def get_preprocessor() -> VietnamesePreprocessor:
    """Get or create the Vietnamese preprocessor singleton."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = VietnamesePreprocessor()
    return _preprocessor
