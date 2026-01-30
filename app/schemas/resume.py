"""
Pydantic V2 Resume Schema with comprehensive validation.

This schema defines the standard JSON format for parsed CVs including:
- Personal information (name, email, phone, etc.)
- Education history
- Work experience
- Skills and projects
- Validation warnings for missing/invalid data
"""

from datetime import date
from typing import Optional, List, Dict, Any

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class SocialLinks(BaseModel):
    """Social media and professional links."""

    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    twitter: Optional[str] = None
    facebook: Optional[str] = None
    other: Dict[str, str] = Field(default_factory=dict)


class Education(BaseModel):
    """Educational background entry."""

    institution: str = Field(..., description="Name of educational institution")
    degree: str = Field(..., description="Degree obtained or pursuing")
    field_of_study: Optional[str] = Field(None, description="Major or specialization")
    start_date: Optional[date] = None
    end_date: Optional[date] = Field(None, description="None if currently studying")
    gpa: Optional[float] = Field(None, ge=0, le=4.0, description="GPA on 4.0 scale")
    achievements: List[str] = Field(default_factory=list)
    location: Optional[str] = None

    @field_validator("gpa", mode="before")
    @classmethod
    def normalize_gpa(cls, v: Any) -> Optional[float]:
        """Normalize GPA to 4.0 scale if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.replace(",", ".").strip()
            try:
                v = float(v)
            except ValueError:
                return None
        # If GPA is on 10.0 scale (common in Vietnam), convert to 4.0
        if v > 4.0:
            return round(v * 0.4, 2)
        return round(v, 2)


class WorkExperience(BaseModel):
    """Work experience entry."""

    company: str = Field(..., description="Company or organization name")
    position: str = Field(..., description="Job title or position")
    start_date: date = Field(..., description="Start date of employment")
    end_date: Optional[date] = Field(None, description="None if current position")
    location: Optional[str] = None
    description: Optional[str] = Field(None, description="Role description")
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list, description="Tech stack used")

    @property
    def is_current(self) -> bool:
        """Check if this is the current position."""
        return self.end_date is None

    @property
    def duration_months(self) -> int:
        """Calculate duration in months."""
        end = self.end_date or date.today()
        months = (end.year - self.start_date.year) * 12
        months += end.month - self.start_date.month
        return max(0, months)

    @model_validator(mode="after")
    def validate_dates(self) -> "WorkExperience":
        """Ensure end_date is after start_date."""
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        return self


class Project(BaseModel):
    """Project entry."""

    name: str = Field(..., description="Project name")
    description: Optional[str] = None
    role: Optional[str] = Field(None, description="Role in the project")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = Field(None, description="Project URL or repository")
    highlights: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    """Certification or license entry."""

    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None
    credential_id: Optional[str] = None
    credential_url: Optional[str] = None


class Language(BaseModel):
    """Language proficiency."""

    language: str
    proficiency: Optional[str] = Field(
        None,
        description="Proficiency level: Native, Fluent, Advanced, Intermediate, Basic",
    )


class ResumeSchema(BaseModel):
    """
    Complete resume/CV schema with validation.

    This is the standard JSON format for all parsed CVs in the system.
    The LLM parser must output data conforming to this schema.
    """

    # Personal Information
    full_name: str = Field(..., description="Full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Phone number with country code")
    date_of_birth: Optional[date] = None
    address: Optional[str] = Field(None, description="Full address or city/country")
    nationality: Optional[str] = None

    # Profile
    summary: Optional[str] = Field(
        None, description="Professional summary or objective"
    )
    headline: Optional[str] = Field(
        None, description="Professional headline (e.g., 'Senior Software Engineer')"
    )

    # Links
    social_links: SocialLinks = Field(default_factory=SocialLinks)

    # Experience & Education
    education: List[Education] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)

    # Skills
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    skills_by_category: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Skills organized by category (e.g., 'Programming': ['Python', 'Java'])",
    )

    # Projects & Certifications
    projects: List[Project] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)

    # Languages
    languages: List[Language] = Field(default_factory=list)

    # Additional
    awards: List[str] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)

    # Validation
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings (missing contact info, etc.)",
    )

    # Metadata
    raw_text: Optional[str] = Field(None, description="Original raw text from CV")
    source_file: Optional[str] = Field(None, description="Original filename")
    parsed_at: Optional[str] = None

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, v: Any) -> Optional[str]:
        """Handle empty or invalid email."""
        if v is None or v == "" or v == "null":
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            # Basic email validation
            if "@" not in v or "." not in v.split("@")[-1]:
                return None
        return v

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: Any) -> Optional[str]:
        """Normalize phone number format."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            # Remove common separators but keep + for country code
            v = v.strip()
            # Remove spaces, dashes, parentheses
            normalized = "".join(c for c in v if c.isdigit() or c == "+")
            if len(normalized) < 8:  # Too short to be valid
                return v  # Return original for manual review
            return normalized
        return str(v)

    @model_validator(mode="after")
    def add_validation_warnings(self) -> "ResumeSchema":
        """Add validation warnings for missing critical information."""
        warnings = list(self.validation_warnings)

        if not self.email:
            warnings.append("Missing Contact Info: Email not found or invalid")
        if not self.phone:
            warnings.append("Missing Contact Info: Phone number not found")
        if not self.full_name or len(self.full_name.strip()) < 2:
            warnings.append("Missing Contact Info: Full name not found or too short")
        if not self.work_experience and not self.education:
            warnings.append(
                "Missing Experience: No work experience or education found"
            )

        self.validation_warnings = warnings
        return self

    @property
    def total_experience_intervals(self) -> List[tuple]:
        """Get work experience intervals for merge algorithm."""
        return [
            (exp.start_date, exp.end_date)
            for exp in self.work_experience
            if exp.start_date
        ]

    def get_all_skills(self) -> List[str]:
        """Get all skills from both flat list and categorized dict."""
        all_skills = set(self.skills)
        for skills_list in self.skills_by_category.values():
            all_skills.update(skills_list)
        # Also extract technologies from work experience and projects
        for exp in self.work_experience:
            all_skills.update(exp.technologies)
        for proj in self.projects:
            all_skills.update(proj.technologies)
        return list(all_skills)

    def to_searchable_text(self) -> str:
        """Generate searchable text representation of the resume."""
        parts = [
            self.full_name,
            self.headline or "",
            self.summary or "",
            " ".join(self.skills),
        ]

        for exp in self.work_experience:
            parts.append(f"{exp.position} at {exp.company}")
            if exp.description:
                parts.append(exp.description)
            parts.extend(exp.responsibilities)
            parts.extend(exp.achievements)

        for edu in self.education:
            parts.append(f"{edu.degree} from {edu.institution}")
            if edu.field_of_study:
                parts.append(edu.field_of_study)

        for proj in self.projects:
            parts.append(proj.name)
            if proj.description:
                parts.append(proj.description)

        return " ".join(filter(None, parts))
