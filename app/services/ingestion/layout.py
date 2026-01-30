"""
Layout Processor for handling two-column and complex CV layouts.

The key insight is that CVs often have two-column layouts, and simple
line-by-line reading would incorrectly merge text from left and right columns.

This processor:
1. Detects column boundaries using text block positions
2. Groups text blocks by column
3. Reads column-wise (left column top-to-bottom, then right column)
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from app.services.ingestion.ocr import OCRBlock

logger = logging.getLogger(__name__)


@dataclass
class Column:
    """Represents a detected column in the layout."""

    x_start: int
    x_end: int
    blocks: List[OCRBlock]

    @property
    def center(self) -> float:
        return (self.x_start + self.x_end) / 2

    @property
    def width(self) -> int:
        return self.x_end - self.x_start


@dataclass
class LayoutAnalysis:
    """Result of layout analysis."""

    is_multi_column: bool
    num_columns: int
    columns: List[Column]
    page_width: int
    page_height: int


class LayoutProcessor:
    """
    Processor for analyzing and handling complex CV layouts.
    
    Handles:
    - Single column layouts
    - Two-column layouts (common in modern CVs)
    - Mixed layouts (header + two columns)
    
    Algorithm:
    1. Cluster text blocks by x-position to detect columns
    2. Determine column boundaries
    3. Assign blocks to columns
    4. Read each column top-to-bottom
    """

    def __init__(
        self,
        column_gap_threshold: float = 0.15,
        min_column_width: float = 0.2,
    ):
        """
        Args:
            column_gap_threshold: Minimum gap between columns as fraction of page width
            min_column_width: Minimum column width as fraction of page width
        """
        self.column_gap_threshold = column_gap_threshold
        self.min_column_width = min_column_width

    def analyze_layout(self, blocks: List[OCRBlock]) -> LayoutAnalysis:
        """
        Analyze the layout of text blocks to detect columns.
        
        Args:
            blocks: List of OCR blocks with position information
            
        Returns:
            LayoutAnalysis with detected columns
        """
        if not blocks:
            return LayoutAnalysis(
                is_multi_column=False,
                num_columns=0,
                columns=[],
                page_width=0,
                page_height=0,
            )

        # Get page dimensions from block positions
        page_width = max(b.bbox[2] for b in blocks)
        page_height = max(b.bbox[3] for b in blocks)

        # Detect columns based on x-center clustering
        columns = self._detect_columns(blocks, page_width)

        return LayoutAnalysis(
            is_multi_column=len(columns) > 1,
            num_columns=len(columns),
            columns=columns,
            page_width=page_width,
            page_height=page_height,
        )

    def _detect_columns(
        self, blocks: List[OCRBlock], page_width: int
    ) -> List[Column]:
        """
        Detect columns by analyzing x-position distribution of blocks.
        
        Uses a simple histogram-based approach:
        1. Create histogram of block x-centers
        2. Find gaps in the histogram that indicate column boundaries
        3. Validate detected columns meet minimum width requirements
        """
        if not blocks:
            return []

        # Get x-centers of all blocks
        x_centers = [b.center_x for b in blocks]

        # Create histogram of x-positions
        num_bins = 50
        hist, bin_edges = self._create_histogram(x_centers, num_bins, page_width)

        # Find column boundaries (gaps in histogram)
        gap_threshold = self.column_gap_threshold * page_width
        column_boundaries = self._find_column_boundaries(
            hist, bin_edges, gap_threshold, page_width
        )

        # Create columns and assign blocks
        columns = []
        for i, (x_start, x_end) in enumerate(column_boundaries):
            column_blocks = [
                b for b in blocks
                if x_start <= b.center_x <= x_end
            ]
            # Sort blocks by y-position (top to bottom)
            column_blocks.sort(key=lambda b: b.center_y)

            if column_blocks:
                columns.append(
                    Column(
                        x_start=int(x_start),
                        x_end=int(x_end),
                        blocks=column_blocks,
                    )
                )

        # Sort columns left to right
        columns.sort(key=lambda c: c.x_start)

        return columns

    def _create_histogram(
        self, x_centers: List[float], num_bins: int, page_width: int
    ) -> Tuple[List[int], List[float]]:
        """Create histogram of x-positions."""
        bin_width = page_width / num_bins
        hist = [0] * num_bins
        bin_edges = [i * bin_width for i in range(num_bins + 1)]

        for x in x_centers:
            bin_idx = min(int(x / bin_width), num_bins - 1)
            hist[bin_idx] += 1

        return hist, bin_edges

    def _find_column_boundaries(
        self,
        hist: List[int],
        bin_edges: List[float],
        gap_threshold: float,
        page_width: int,
    ) -> List[Tuple[float, float]]:
        """Find column boundaries based on histogram gaps."""
        min_column_width_px = self.min_column_width * page_width

        # Find continuous regions with content
        in_column = False
        column_start = 0
        boundaries = []

        for i, count in enumerate(hist):
            if count > 0 and not in_column:
                # Start of a new column
                column_start = bin_edges[i]
                in_column = True
            elif count == 0 and in_column:
                # Potential end of column
                column_end = bin_edges[i]

                # Check if gap is significant
                gap_size = self._get_gap_size(hist, i, bin_edges)
                if gap_size >= gap_threshold:
                    # Valid column boundary
                    if column_end - column_start >= min_column_width_px:
                        boundaries.append((column_start, column_end))
                    in_column = False

        # Handle last column
        if in_column:
            column_end = bin_edges[-1]
            if column_end - column_start >= min_column_width_px:
                boundaries.append((column_start, column_end))

        # If no valid columns found, treat entire page as one column
        if not boundaries:
            boundaries = [(0, page_width)]

        return boundaries

    def _get_gap_size(
        self, hist: List[int], start_idx: int, bin_edges: List[float]
    ) -> float:
        """Calculate the size of a gap starting at start_idx."""
        gap_end = start_idx
        while gap_end < len(hist) and hist[gap_end] == 0:
            gap_end += 1

        if gap_end >= len(hist):
            return 0

        return bin_edges[gap_end] - bin_edges[start_idx]

    def process_two_column(self, blocks: List[OCRBlock]) -> str:
        """
        Process blocks assuming two-column layout.
        
        Reads left column top-to-bottom, then right column top-to-bottom.
        This prevents mixing text from adjacent columns.
        
        Args:
            blocks: OCR blocks with position information
            
        Returns:
            Combined text with proper column ordering
        """
        analysis = self.analyze_layout(blocks)

        if not analysis.is_multi_column:
            # Single column - just sort by y-position
            sorted_blocks = sorted(blocks, key=lambda b: (b.center_y, b.center_x))
            return self._blocks_to_text(sorted_blocks)

        # Multi-column - process each column separately
        text_parts = []
        for column in analysis.columns:
            column_text = self._blocks_to_text(column.blocks)
            text_parts.append(column_text)

        return "\n\n".join(text_parts)

    def _blocks_to_text(self, blocks: List[OCRBlock]) -> str:
        """Convert blocks to text, grouping by lines."""
        if not blocks:
            return ""

        # Group blocks into lines based on y-position
        lines = self._group_into_lines(blocks)

        text_lines = []
        for line_blocks in lines:
            # Sort blocks in line by x-position (left to right)
            sorted_line = sorted(line_blocks, key=lambda b: b.center_x)
            line_text = " ".join(b.text for b in sorted_line)
            text_lines.append(line_text)

        return "\n".join(text_lines)

    def _group_into_lines(
        self, blocks: List[OCRBlock], tolerance: float = 0.5
    ) -> List[List[OCRBlock]]:
        """
        Group blocks into lines based on y-position.
        
        Blocks with similar y-centers are considered part of the same line.
        """
        if not blocks:
            return []

        # Sort by y-position
        sorted_blocks = sorted(blocks, key=lambda b: b.center_y)

        lines = []
        current_line = [sorted_blocks[0]]
        current_y = sorted_blocks[0].center_y

        for block in sorted_blocks[1:]:
            # Calculate line height tolerance
            avg_height = sum(b.height for b in current_line) / len(current_line)
            threshold = avg_height * tolerance

            if abs(block.center_y - current_y) <= threshold:
                # Same line
                current_line.append(block)
            else:
                # New line
                lines.append(current_line)
                current_line = [block]
                current_y = block.center_y

        # Add last line
        if current_line:
            lines.append(current_line)

        return lines

    def merge_columns_text(self, columns: List[Column]) -> str:
        """
        Merge text from multiple columns.
        
        Args:
            columns: List of detected columns
            
        Returns:
            Combined text with column markers
        """
        if not columns:
            return ""

        text_parts = []
        for i, column in enumerate(columns):
            column_text = self._blocks_to_text(column.blocks)
            text_parts.append(f"[Column {i + 1}]\n{column_text}")

        return "\n\n".join(text_parts)

    def get_header_blocks(
        self,
        blocks: List[OCRBlock],
        header_fraction: float = 0.15,
    ) -> List[OCRBlock]:
        """
        Get blocks that appear to be in the header area.
        
        Headers often span the full width and contain contact info.
        
        Args:
            blocks: All OCR blocks
            header_fraction: Fraction of page height to consider as header
            
        Returns:
            Blocks in the header area
        """
        if not blocks:
            return []

        page_height = max(b.bbox[3] for b in blocks)
        header_threshold = page_height * header_fraction

        return [b for b in blocks if b.center_y <= header_threshold]
