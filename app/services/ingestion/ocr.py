"""
OCR Service using PaddleOCR for text extraction with layout awareness.

Handles:
- Image-based PDFs
- Scanned documents
- Multi-language support (Vietnamese + English)
- Bounding box extraction for layout analysis
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class OCRBlock:
    """Represents a text block with position information."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_x: float
    center_y: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class OCRService:
    """
    OCR service using PaddleOCR for layout-aware text extraction.
    
    Features:
    - Vietnamese and English text recognition
    - Bounding box extraction for layout analysis
    - Confidence scoring
    - GPU acceleration (optional)
    """

    def __init__(self):
        self._ocr = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of PaddleOCR (heavy import)."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ocr_lang,
                use_gpu=settings.ocr_use_gpu,
                show_log=False,
                # Optimization settings
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                rec_batch_num=16,
            )
            self._initialized = True
            logger.info(
                f"PaddleOCR initialized with lang={settings.ocr_lang}, "
                f"gpu={settings.ocr_use_gpu}"
            )
        except ImportError:
            logger.error("PaddleOCR not installed. Please install paddleocr package.")
            raise

    def extract_from_image(self, image_path: str) -> List[OCRBlock]:
        """
        Extract text blocks from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of OCRBlock with text, confidence, and position
        """
        self._lazy_init()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = self._ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            logger.warning(f"No text detected in {image_path}")
            return []

        return self._parse_ocr_result(result[0])

    def extract_from_pdf(self, pdf_path: str) -> List[List[OCRBlock]]:
        """
        Extract text blocks from each page of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of OCRBlock lists, one per page
        """
        self._lazy_init()

        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.error("pdf2image not installed")
            raise

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300)
        all_blocks = []

        for page_num, image in enumerate(images, 1):
            # Save temporary image
            temp_path = f"/tmp/cv_page_{page_num}.png"
            image.save(temp_path, "PNG")

            try:
                blocks = self.extract_from_image(temp_path)
                all_blocks.append(blocks)
                logger.info(
                    f"Page {page_num}: extracted {len(blocks)} text blocks"
                )
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return all_blocks

    def extract_text_only(self, file_path: str) -> str:
        """
        Extract plain text from image or PDF (without layout info).
        
        This is a convenience method that combines all text blocks.
        For layout-aware extraction, use extract_from_image/extract_from_pdf.
        """
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            pages = self.extract_from_pdf(file_path)
            texts = []
            for page_blocks in pages:
                # Sort blocks by position (top to bottom, left to right)
                sorted_blocks = sorted(
                    page_blocks, key=lambda b: (b.center_y, b.center_x)
                )
                page_text = " ".join(b.text for b in sorted_blocks)
                texts.append(page_text)
            return "\n\n".join(texts)
        else:
            blocks = self.extract_from_image(file_path)
            sorted_blocks = sorted(
                blocks, key=lambda b: (b.center_y, b.center_x)
            )
            return " ".join(b.text for b in sorted_blocks)

    def _parse_ocr_result(self, result: List) -> List[OCRBlock]:
        """Parse PaddleOCR result into OCRBlock objects."""
        blocks = []

        for line in result:
            if not line or len(line) < 2:
                continue

            bbox_points, (text, confidence) = line

            # Convert polygon to bounding box
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            bbox = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            blocks.append(
                OCRBlock(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                )
            )

        return blocks

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if a PDF is scanned (image-based) or has extractable text.
        
        Returns True if the PDF appears to be scanned and needs OCR.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            total_chars = 0

            for page in doc:
                text = page.get_text()
                total_chars += len(text.strip())

            doc.close()

            # If very little text is extractable, likely a scanned PDF
            return total_chars < 100

        except Exception as e:
            logger.warning(f"Error checking PDF type: {e}")
            return True  # Assume scanned if we can't determine

    def extract_text_hybrid(self, pdf_path: str) -> str:
        """
        Hybrid extraction: use native text if available, OCR if scanned.
        
        This is the recommended method for processing CVs.
        """
        if self.is_scanned_pdf(pdf_path):
            logger.info(f"PDF appears scanned, using OCR: {pdf_path}")
            return self.extract_text_only(pdf_path)
        else:
            logger.info(f"PDF has native text, extracting directly: {pdf_path}")
            return self._extract_native_text(pdf_path)

    def _extract_native_text(self, pdf_path: str) -> str:
        """Extract text from PDF with native text layer."""
        try:
            import fitz

            doc = fitz.open(pdf_path)
            texts = []

            for page in doc:
                texts.append(page.get_text())

            doc.close()
            return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Error extracting native text: {e}")
            # Fallback to OCR
            return self.extract_text_only(pdf_path)
