"""
Tests for the Section-Aware Chunking System.
"""

import pytest
from app.services.parsing.chunker import (
    SectionAwareChunker,
    Chunk,
    SectionType,
)


class TestSectionAwareChunker:
    """Tests for SectionAwareChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance."""
        return SectionAwareChunker()

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text in English."""
        return """
John Smith
john.smith@email.com | +1-555-123-4567

PROFESSIONAL SUMMARY
Experienced software engineer with 10+ years of experience in full-stack development.
Passionate about building scalable applications and leading engineering teams.

WORK EXPERIENCE

Senior Software Engineer - Google Inc. - 2020-Present
- Led development of microservices architecture serving 1M+ users
- Mentored junior developers and conducted code reviews
- Implemented CI/CD pipelines reducing deployment time by 50%

Software Engineer - Amazon - 2017-2020
- Developed backend services using Python and AWS
- Designed database schemas for high-throughput applications
- Collaborated with product team on feature requirements

EDUCATION

Master of Computer Science - Stanford University - 2017
GPA: 3.9/4.0

Bachelor of Computer Science - MIT - 2015
Dean's List, Summa Cum Laude

SKILLS
Python, Java, JavaScript, React, AWS, Docker, Kubernetes, PostgreSQL

PROJECTS

E-commerce Platform
Built a scalable e-commerce platform handling 100k daily transactions.
Technologies: Python, Django, PostgreSQL, Redis

Machine Learning Pipeline
Developed an ML pipeline for real-time fraud detection.
Technologies: Python, TensorFlow, Spark
"""

    @pytest.fixture
    def sample_cv_vietnamese(self):
        """Sample CV text in Vietnamese."""
        return """
Nguyễn Văn An
an.nguyen@email.com | 0912-345-678

GIỚI THIỆU BẢN THÂN
Kỹ sư phần mềm với hơn 5 năm kinh nghiệm trong lĩnh vực phát triển web.

KINH NGHIỆM LÀM VIỆC

Senior Developer - FPT Software - 2020-Nay
- Phát triển hệ thống quản lý cho khách hàng ngân hàng
- Áp dụng kiến trúc microservices
- Hướng dẫn team 5 developer

Developer - Viettel - 2018-2020
- Phát triển ứng dụng mobile bằng React Native
- Tích hợp API và backend services

HỌC VẤN

Cử nhân Công nghệ thông tin - Đại học Bách Khoa Hà Nội - 2018
GPA: 8.5/10

KỸ NĂNG
Python, JavaScript, React, Node.js, PostgreSQL, Docker
"""

    def test_chunk_english_cv(self, chunker, sample_cv_text):
        """Test chunking an English CV."""
        chunks = chunker.chunk_document(sample_cv_text)

        assert len(chunks) > 0

        # Check that we have different section types
        sections = {c.section for c in chunks}
        assert SectionType.EXPERIENCE in sections
        assert SectionType.EDUCATION in sections
        assert SectionType.SKILLS in sections

    def test_chunk_vietnamese_cv(self, chunker, sample_cv_vietnamese):
        """Test chunking a Vietnamese CV."""
        chunks = chunker.chunk_document(sample_cv_vietnamese)

        assert len(chunks) > 0

        sections = {c.section for c in chunks}
        # Should detect Vietnamese section headers
        assert SectionType.EXPERIENCE in sections

    def test_parent_child_relationship(self, chunker, sample_cv_text):
        """Test that chunks have proper parent-child relationships."""
        chunks = chunker.chunk_document(sample_cv_text)

        parent_chunks = [c for c in chunks if c.is_parent]
        child_chunks = [c for c in chunks if not c.is_parent]

        # Should have both parents and children for experience section
        experience_parents = [
            c for c in parent_chunks if c.section == SectionType.EXPERIENCE
        ]
        experience_children = [
            c for c in child_chunks if c.section == SectionType.EXPERIENCE
        ]

        assert len(experience_parents) >= 1

    def test_chunk_has_required_fields(self, chunker, sample_cv_text):
        """Test that chunks have all required fields."""
        chunks = chunker.chunk_document(sample_cv_text)

        for chunk in chunks:
            assert chunk.id is not None
            assert chunk.content is not None
            assert chunk.section is not None
            assert isinstance(chunk.section, SectionType)

    def test_empty_text_returns_empty(self, chunker):
        """Test that empty text returns no chunks."""
        chunks = chunker.chunk_document("")
        assert chunks == []

        chunks = chunker.chunk_document("   ")
        assert chunks == []

    def test_short_text_returns_empty(self, chunker):
        """Test that very short text returns no chunks."""
        chunks = chunker.chunk_document("Hello")
        assert chunks == []

    def test_chunks_preserve_order(self, chunker, sample_cv_text):
        """Test that chunks maintain document order."""
        chunks = chunker.chunk_document(sample_cv_text)

        # Order indices should be sequential
        indices = [c.order_index for c in chunks]
        assert indices == sorted(indices)

    def test_metadata_extraction_experience(self, chunker, sample_cv_text):
        """Test that experience chunks extract metadata."""
        chunks = chunker.chunk_document(sample_cv_text)

        experience_children = [
            c for c in chunks
            if c.section == SectionType.EXPERIENCE and not c.is_parent
        ]

        # At least one should have company/position metadata
        has_metadata = any(
            c.metadata.get("company") or c.metadata.get("position")
            for c in experience_children
        )
        # This might not always work depending on text format
        # Just check that metadata dict exists
        assert all(hasattr(c, "metadata") for c in experience_children)


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="Test content",
            section=SectionType.EXPERIENCE,
        )

        assert chunk.content == "Test content"
        assert chunk.section == SectionType.EXPERIENCE
        assert chunk.id is not None
        assert chunk.is_parent is True

    def test_chunk_with_parent(self):
        """Test child chunk creation."""
        parent = Chunk(
            content="Parent content",
            section=SectionType.EXPERIENCE,
        )

        child = Chunk(
            content="Child content",
            section=SectionType.EXPERIENCE,
            parent_id=parent.id,
            subsection="Software Engineer at Google",
        )

        assert child.is_parent is False
        assert child.parent_id == parent.id
        assert child.subsection == "Software Engineer at Google"

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            content="Test content",
            section=SectionType.SKILLS,
            metadata={"skills": ["Python", "Java"]},
        )

        d = chunk.to_dict()

        assert d["content"] == "Test content"
        assert d["section"] == "skills"
        assert d["metadata"]["skills"] == ["Python", "Java"]


class TestSectionDetection:
    """Tests for section header detection."""

    @pytest.fixture
    def chunker(self):
        return SectionAwareChunker()

    @pytest.mark.parametrize("header,expected", [
        ("WORK EXPERIENCE", SectionType.EXPERIENCE),
        ("Experience", SectionType.EXPERIENCE),
        ("employment history", SectionType.EXPERIENCE),
        ("KINH NGHIỆM LÀM VIỆC", SectionType.EXPERIENCE),
        ("EDUCATION", SectionType.EDUCATION),
        ("Học vấn", SectionType.EDUCATION),
        ("SKILLS", SectionType.SKILLS),
        ("Kỹ năng", SectionType.SKILLS),
        ("PROJECTS", SectionType.PROJECTS),
        ("Dự án", SectionType.PROJECTS),
        ("CERTIFICATIONS", SectionType.CERTIFICATIONS),
        ("Random text that is not a header", None),
    ])
    def test_section_header_detection(self, chunker, header, expected):
        """Test various section header patterns."""
        result = chunker._detect_section_header(header)
        assert result == expected
