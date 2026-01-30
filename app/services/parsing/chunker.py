"""
Section-Aware Chunker for intelligent CV splitting.

Implements a two-level Parent-Child chunking strategy:
- Level 1 (Parent): Major sections (Experience, Education, Projects, Skills)
- Level 2 (Child): Individual items within sections (specific jobs, degrees)

This approach:
1. Preserves document structure for better understanding
2. Enables granular search (find specific job experiences)
3. Maintains context through parent-child relationships
"""

import re
import uuid
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SectionType(str, Enum):
    """Types of CV sections."""

    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"
    AWARDS = "awards"
    REFERENCES = "references"
    OTHER = "other"


@dataclass
class Chunk:
    """
    Represents a chunk of CV content.
    
    Attributes:
        id: Unique identifier
        content: Raw text content
        section: Section type (Experience, Education, etc.)
        subsection: Specific item (job title, degree, project name)
        parent_id: ID of parent chunk (None for parent chunks)
        metadata: Additional context (company, dates, etc.)
        order_index: Order within the document
    """

    content: str
    section: SectionType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subsection: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    order_index: int = 0

    @property
    def is_parent(self) -> bool:
        """Check if this is a parent chunk."""
        return self.parent_id is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "section": self.section.value,
            "subsection": self.subsection,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "order_index": self.order_index,
        }


class SectionAwareChunker:
    """
    Intelligent chunker that splits CVs by sections and items.
    
    Instead of fixed-size chunking, this chunker:
    1. Detects section headers using heuristics and patterns
    2. Creates parent chunks for each major section
    3. Creates child chunks for individual items (jobs, projects)
    4. Preserves hierarchy through parent_id references
    """

    # Section header patterns (English and Vietnamese)
    SECTION_PATTERNS = {
        SectionType.EXPERIENCE: [
            r"(?i)^(?:work\s*)?experience",
            r"(?i)^employment\s*(?:history)?",
            r"(?i)^professional\s*experience",
            r"(?i)^career\s*(?:history)?",
            r"(?i)^kinh\s*nghiệm(?:\s*làm\s*việc)?",
            r"(?i)^quá\s*trình\s*(?:làm\s*việc|công\s*tác)",
        ],
        SectionType.EDUCATION: [
            r"(?i)^education(?:al\s*background)?",
            r"(?i)^academic\s*(?:background|qualifications?)",
            r"(?i)^học\s*vấn",
            r"(?i)^trình\s*độ\s*(?:học\s*vấn)?",
            r"(?i)^bằng\s*cấp",
        ],
        SectionType.SKILLS: [
            r"(?i)^(?:technical\s*)?skills?",
            r"(?i)^competenc(?:y|ies)",
            r"(?i)^expertise",
            r"(?i)^kỹ\s*năng",
            r"(?i)^chuyên\s*(?:môn|ngành)",
        ],
        SectionType.PROJECTS: [
            r"(?i)^projects?",
            r"(?i)^(?:key\s*)?(?:projects?|initiatives?)",
            r"(?i)^(?:personal\s*)?projects?",
            r"(?i)^side\s*projects?",
            r"(?i)^portfolio",
            r"(?i)^dự\s*án(?:\s*(?:cá\s*nhân|tiêu\s*biểu))?",
            r"(?i)^các\s*dự\s*án",
            r"(?i)^đề\s*tài(?:\s*nghiên\s*cứu)?",
            r"(?i)^luận\s*văn",
            r"(?i)^(?:final\s*year\s*)?(?:thesis|capstone)",
            r"(?i)^sản\s*phẩm",
        ],
        SectionType.CERTIFICATIONS: [
            r"(?i)^certifications?(?:\s*&\s*licenses?)?",
            r"(?i)^licenses?\s*(?:&\s*)?certifications?",
            r"(?i)^(?:professional\s*)?qualifications?",
            r"(?i)^chứng\s*chỉ",
            r"(?i)^bằng\s*cấp(?:\s*chuyên\s*môn)?",
        ],
        SectionType.SUMMARY: [
            r"(?i)^(?:professional\s*)?summary",
            r"(?i)^(?:career\s*)?objective",
            r"(?i)^profile(?:\s*summary)?",
            r"(?i)^about\s*(?:me)?",
            r"(?i)^giới\s*thiệu(?:\s*bản\s*thân)?",
            r"(?i)^mục\s*tiêu(?:\s*nghề\s*nghiệp)?",
            r"(?i)^tóm\s*tắt",
        ],
        SectionType.LANGUAGES: [
            r"(?i)^languages?",
            r"(?i)^language\s*skills?",
            r"(?i)^ngôn\s*ngữ",
            r"(?i)^ngoại\s*ngữ",
        ],
        SectionType.AWARDS: [
            r"(?i)^awards?(?:\s*&\s*(?:honors?|achievements?))?",
            r"(?i)^(?:honors?|achievements?)(?:\s*&\s*awards?)?",
            r"(?i)^giải\s*thưởng",
            r"(?i)^thành\s*tích",
        ],
        SectionType.REFERENCES: [
            r"(?i)^references?",
            r"(?i)^người\s*tham\s*chiếu",
            r"(?i)^người\s*giới\s*thiệu",
        ],
    }

    # Patterns for detecting individual items within sections
    ITEM_PATTERNS = {
        SectionType.EXPERIENCE: [
            # Company - Position - Date range pattern
            r"(?P<position>.+?)(?:\s*[-–|]\s*|\s+at\s+|\s+tại\s+)(?P<company>.+?)(?:\s*[-–|]\s*)?(?P<dates>\d{1,2}[\/\-]\d{2,4}.+\d{1,2}[\/\-]\d{2,4}|\d{4}\s*[-–]\s*(?:\d{4}|present|current|nay|hiện\s*tại))?",
            # Just company name followed by dates
            r"^(?P<company>[A-Z][^•\n]+?)(?:\s*[-–|]\s*)(?P<dates>\d{4}\s*[-–]\s*(?:\d{4}|Present|Current|Nay))",
        ],
        SectionType.EDUCATION: [
            # Degree - School - Dates
            r"(?P<degree>(?:Bachelor|Master|Ph\.?D|MBA|B\.?S\.?|M\.?S\.?|Cử\s*nhân|Thạc\s*sĩ|Tiến\s*sĩ|Kỹ\s*sư).+?)(?:\s*[-–|]\s*|\s+at\s+|\s+tại\s+)(?P<school>.+?)(?:\s*[-–|]\s*)?(?P<dates>\d{4}.*)?",
            # School name pattern
            r"^(?P<school>(?:University|College|Institute|Đại\s*học|Học\s*viện|Cao\s*đẳng).+?)(?:\s*[-–|]\s*)?(?P<dates>\d{4}.*)?",
        ],
        SectionType.PROJECTS: [
            # Project name with description
            r"^(?P<name>[^•\n]{3,50}?)(?:\s*[-–:]\s*)(?P<description>.+)",
        ],
    }

    def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 2000):
        """
        Args:
            min_chunk_size: Minimum characters for a valid chunk
            max_chunk_size: Maximum characters before splitting further
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_document(self, text: str) -> List[Chunk]:
        """
        Chunk a CV document using section-aware approach.
        
        Args:
            text: Full CV text
            
        Returns:
            List of Chunk objects with parent-child relationships
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # Step 1: Split by major sections
        sections = self._detect_sections(text)

        # Step 2: Create parent and child chunks for each section
        all_chunks = []
        order_index = 0

        for section_type, section_content in sections:
            # Create parent chunk for the section
            parent_chunk = Chunk(
                content=section_content,
                section=section_type,
                order_index=order_index,
            )
            order_index += 1

            # Create child chunks for items within the section
            children = self._create_child_chunks(
                section_content, section_type, parent_chunk.id
            )

            if children:
                # Add parent and children
                all_chunks.append(parent_chunk)
                for child in children:
                    child.order_index = order_index
                    order_index += 1
                    all_chunks.append(child)
            else:
                # No children, just add parent if it has enough content
                if len(section_content.strip()) >= self.min_chunk_size:
                    all_chunks.append(parent_chunk)

        logger.info(
            f"Created {len(all_chunks)} chunks from document "
            f"({sum(1 for c in all_chunks if c.is_parent)} parents, "
            f"{sum(1 for c in all_chunks if not c.is_parent)} children)"
        )

        return all_chunks

    def _detect_sections(self, text: str) -> List[tuple]:
        """
        Detect and split text into major sections.
        
        Returns list of (SectionType, content) tuples.
        """
        sections = []
        lines = text.split("\n")

        current_section = SectionType.OTHER
        current_content = []

        for line in lines:
            # Check if this line is a section header
            detected_section = self._detect_section_header(line)

            if detected_section and detected_section != current_section:
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections.append((current_section, content))

                # Start new section
                current_section = detected_section
                current_content = [line]
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections.append((current_section, content))

        return sections

    def _detect_section_header(self, line: str) -> Optional[SectionType]:
        """Check if a line is a section header and return its type."""
        line = line.strip()

        if not line or len(line) > 100:  # Headers are usually short
            return None

        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line):
                    return section_type

        return None

    def _create_child_chunks(
        self,
        section_content: str,
        section_type: SectionType,
        parent_id: str,
    ) -> List[Chunk]:
        """
        Create child chunks for individual items within a section.
        
        For Experience: each job position
        For Education: each degree/school
        For Projects: each project
        """
        if section_type not in [
            SectionType.EXPERIENCE,
            SectionType.EDUCATION,
            SectionType.PROJECTS,
        ]:
            return []

        children = []
        items = self._split_section_into_items(section_content, section_type)

        for item_content, item_metadata in items:
            if len(item_content.strip()) >= self.min_chunk_size:
                child = Chunk(
                    content=item_content,
                    section=section_type,
                    subsection=item_metadata.get("title", None),
                    parent_id=parent_id,
                    metadata=item_metadata,
                )
                children.append(child)

        return children

    def _split_section_into_items(
        self, content: str, section_type: SectionType
    ) -> List[tuple]:
        """
        Split section content into individual items.
        
        Returns list of (content, metadata) tuples.
        """
        if section_type == SectionType.EXPERIENCE:
            return self._split_experience_section(content)
        elif section_type == SectionType.EDUCATION:
            return self._split_education_section(content)
        elif section_type == SectionType.PROJECTS:
            return self._split_projects_section(content)
        return []

    def _split_experience_section(self, content: str) -> List[tuple]:
        """Split work experience section into individual job entries."""
        items = []

        # Pattern for job entries (company names often start with capital letters)
        # or job titles followed by company
        job_pattern = r"(?:^|\n\n)([A-Z][^\n]*(?:\n(?![A-Z])[^\n]+)*)"

        # Try to find job entries by looking for date patterns
        date_pattern = r"(\d{1,2}[\/\-]\d{2,4}|\d{4})\s*[-–]\s*(\d{1,2}[\/\-]\d{2,4}|\d{4}|present|current|nay|hiện\s*tại)"

        # Split by blank lines and check each block
        blocks = re.split(r"\n{2,}", content)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Extract metadata if possible
            metadata = {}
            lines = block.split("\n")

            if lines:
                first_line = lines[0].strip()
                # Try to extract position and company
                if " - " in first_line or " | " in first_line or " at " in first_line.lower():
                    parts = re.split(r"\s*[-|]\s*|\s+at\s+", first_line, 1)
                    if len(parts) >= 2:
                        metadata["position"] = parts[0].strip()
                        metadata["company"] = parts[1].strip()
                        metadata["title"] = f"{parts[0].strip()} at {parts[1].strip()}"

                # Try to extract dates
                date_match = re.search(date_pattern, block, re.IGNORECASE)
                if date_match:
                    metadata["start_date"] = date_match.group(1)
                    metadata["end_date"] = date_match.group(2)
                    metadata["duration"] = f"{date_match.group(1)} - {date_match.group(2)}"

            items.append((block, metadata))

        return items

    def _split_education_section(self, content: str) -> List[tuple]:
        """Split education section into individual degree/school entries."""
        items = []
        blocks = re.split(r"\n{2,}", content)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            metadata = {}
            lines = block.split("\n")

            if lines:
                first_line = lines[0].strip()

                # Try to detect degree and school
                degree_keywords = [
                    "bachelor", "master", "ph.d", "mba", "b.s", "m.s",
                    "cử nhân", "thạc sĩ", "tiến sĩ", "kỹ sư"
                ]
                school_keywords = [
                    "university", "college", "institute",
                    "đại học", "học viện", "cao đẳng"
                ]

                # Check for degree
                for keyword in degree_keywords:
                    if keyword.lower() in first_line.lower():
                        metadata["degree"] = first_line
                        break

                # Check for school
                for keyword in school_keywords:
                    if keyword.lower() in first_line.lower():
                        metadata["school"] = first_line
                        break

                metadata["title"] = first_line

            items.append((block, metadata))

        return items

    def _split_projects_section(self, content: str) -> List[tuple]:
        """Split projects section into individual project entries."""
        items = []

        # Projects are often separated by blank lines or bullet points
        # First try splitting by blank lines
        blocks = re.split(r"\n{2,}", content)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            metadata = {}
            lines = block.split("\n")

            if lines:
                first_line = lines[0].strip()
                # Remove bullet points
                first_line = re.sub(r"^[•\-\*]\s*", "", first_line)
                metadata["name"] = first_line
                metadata["title"] = first_line

            items.append((block, metadata))

        return items

    def get_section_summary(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Get summary of chunks by section type."""
        summary = {}
        for chunk in chunks:
            section = chunk.section.value
            summary[section] = summary.get(section, 0) + 1
        return summary
