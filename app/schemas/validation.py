"""Validation and processing status schemas."""

from enum import Enum
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field


class ProcessingStage(str, Enum):
    """CV processing pipeline stages."""

    UPLOADED = "uploaded"
    OCR_PROCESSING = "ocr_processing"
    LAYOUT_ANALYSIS = "layout_analysis"
    TEXT_PREPROCESSING = "text_preprocessing"
    LLM_PARSING = "llm_parsing"
    EVALUATING = "evaluating"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationSeverity(str, Enum):
    """Severity level for validation warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationWarning(BaseModel):
    """Individual validation warning."""

    code: str = Field(..., description="Warning code (e.g., 'MISSING_EMAIL')")
    message: str = Field(..., description="Human-readable warning message")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    field: Optional[str] = Field(None, description="Field that triggered the warning")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class CVProcessingStatus(BaseModel):
    """Status of CV processing job."""

    task_id: str = Field(..., description="Celery task ID")
    candidate_id: Optional[str] = Field(None, description="Candidate ID if created")
    filename: str
    stage: ProcessingStage = Field(default=ProcessingStage.UPLOADED)
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    
    # Results
    is_complete: bool = False
    is_failed: bool = False
    error_message: Optional[str] = None
    
    # Validation
    validation_warnings: List[ValidationWarning] = Field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None

    @property
    def has_critical_warnings(self) -> bool:
        """Check if there are any error-level warnings."""
        return any(w.severity == ValidationSeverity.ERROR for w in self.validation_warnings)


class CVUploadResponse(BaseModel):
    """Response after CV upload."""

    task_id: str = Field(..., description="Task ID to check processing status")
    filename: str
    message: str = "CV uploaded and processing started"
    status_url: str = Field(..., description="URL to check processing status")


class CandidateResponse(BaseModel):
    """Candidate data response."""

    id: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    total_experience_years: Optional[float] = None
    top_skills: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    created_at: datetime
    
    # Full resume data available on detail view
    resume_data: Optional[dict] = None


class CandidateListResponse(BaseModel):
    """Paginated list of candidates."""

    total: int
    page: int
    page_size: int
    candidates: List[CandidateResponse]
