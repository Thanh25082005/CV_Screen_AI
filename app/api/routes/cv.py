"""
CV Upload and Processing API Routes.

Endpoints for:
- Uploading CV files (PDF, DOCX, images)
- Checking processing status
- Getting parsed CV data
"""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.api.deps import get_settings_dep, get_async_db
from app.config import Settings
from app.schemas.validation import CVUploadResponse, CVProcessingStatus, ProcessingStage
from app.workers.tasks import process_cv_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cv", tags=["CV Management"])


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload", response_model=CVUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_cv(
    file: UploadFile = File(..., description="CV file (PDF, DOCX, or image)"),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Upload a CV file for processing.
    
    The file is saved and a background task is started to:
    1. Extract text (OCR if needed)
    2. Parse with LLM
    3. Chunk and embed
    4. Store in database
    
    Returns a task ID to track progress.
    """
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported: {ALLOWED_EXTENSIONS}",
        )

    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.upload_dir, safe_filename)

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except IOError as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file",
        )

    # Queue processing task
    task = process_cv_task.delay(
        file_path=file_path,
        original_filename=file.filename,
    )

    logger.info(f"CV upload started: {file.filename} -> task {task.id}")

    return CVUploadResponse(
        task_id=task.id,
        filename=file.filename,
        message="CV uploaded and processing started",
        status_url=f"/api/v1/cv/status/{task.id}",
    )


@router.get("/status/{task_id}", response_model=CVProcessingStatus)
async def get_processing_status(task_id: str):
    """
    Get the processing status of an uploaded CV.
    
    Returns current stage, progress, and any errors.
    """
    from app.core.celery_app import celery_app
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    def safe_get_stage(stage_name: str) -> ProcessingStage:
        """Safely convert string to ProcessingStage enum."""
        try:
            return ProcessingStage(stage_name)
        except (ValueError, KeyError):
            logger.warning(f"Unknown processing stage: {stage_name}")
            return ProcessingStage.UPLOADED

    if result.state == "PENDING":
        return CVProcessingStatus(
            task_id=task_id,
            filename="",
            stage=ProcessingStage.UPLOADED,
            progress=0,
        )
    elif result.state == "STARTED":
        info = result.info or {}
        if not isinstance(info, dict): info = {}
        return CVProcessingStatus(
            task_id=task_id,
            filename=info.get("filename", ""),
            stage=safe_get_stage(info.get("stage", "uploaded")),
            progress=info.get("progress", 10),
        )
    elif result.state == "PROGRESS":
        info = result.info or {}
        if not isinstance(info, dict): info = {}
        return CVProcessingStatus(
            task_id=task_id,
            filename=info.get("filename", ""),
            stage=safe_get_stage(info.get("stage", "uploaded")),
            progress=info.get("progress", 0),
        )
    elif result.state == "SUCCESS":
        info = result.result or {}
        if not isinstance(info, dict): info = {}
        return CVProcessingStatus(
            task_id=task_id,
            candidate_id=info.get("candidate_id"),
            filename=info.get("filename", ""),
            stage=ProcessingStage.COMPLETED,
            progress=100,
            is_complete=True,
            validation_warnings=info.get("validation_warnings", []),
        )
    elif result.state == "FAILURE":
        return CVProcessingStatus(
            task_id=task_id,
            filename="",
            stage=ProcessingStage.FAILED,
            progress=0,
            is_failed=True,
            error_message=str(result.result) if result.result else "Processing failed",
        )
    else:
        return CVProcessingStatus(
            task_id=task_id,
            filename="",
            stage=ProcessingStage.UPLOADED,
            progress=0,
        )


@router.get("/{candidate_id}")
async def get_cv_details(
    candidate_id: str,
    db=Depends(get_async_db),
):
    """
    Get the parsed CV data for a candidate.
    
    Returns the full ResumeSchema data.
    """
    from sqlalchemy import select
    from app.models.candidate import Candidate

    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    return {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "email": candidate.email,
        "phone": candidate.phone,
        "headline": candidate.headline,
        "summary": candidate.summary,
        "total_experience_years": candidate.total_experience_years,
        "top_skills": candidate.top_skills,
        "validation_warnings": candidate.validation_warnings,
        "resume_data": candidate.raw_resume,
        "created_at": candidate.created_at,
    }


@router.get("/{candidate_id}/chunks")
async def get_cv_chunks(
    candidate_id: str,
    section: Optional[str] = None,
    db=Depends(get_async_db),
):
    """
    Get the chunks extracted from a candidate's CV.
    
    Optionally filter by section (experience, education, projects, etc.).
    """
    from sqlalchemy import select
    from app.models.candidate import Chunk

    query = select(Chunk).where(Chunk.candidate_id == candidate_id)

    if section:
        query = query.where(Chunk.section == section)

    query = query.order_by(Chunk.order_index)

    result = await db.execute(query)
    chunks = result.scalars().all()

    return {
        "candidate_id": candidate_id,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "section": c.section,
                "subsection": c.subsection,
                "content": c.content,
                "enriched_content": c.metadata.get("enriched_content") if c.metadata else None,
                "metadata": c.metadata,
                "is_parent": c.parent_id is None,
            }
            for c in chunks
        ],
    }


@router.delete("/{candidate_id}")
async def delete_cv(
    candidate_id: str,
    db=Depends(get_async_db),
):
    """
    Delete a candidate and all associated data.
    """
    from sqlalchemy import select, delete
    from app.models.candidate import Candidate

    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate not found: {candidate_id}",
        )

    await db.execute(
        delete(Candidate).where(Candidate.id == candidate_id)
    )
    await db.commit()

    return {"message": f"Candidate {candidate_id} deleted successfully"}


# -----------------------------------------------------------------------------
# Batch Processing
# -----------------------------------------------------------------------------

from pydantic import BaseModel

class ScanRequest(BaseModel):
    directory_path: str = "./public_cvs"
    drive_url: Optional[str] = None


@router.post("/scan")
async def scan_directory(
    request: ScanRequest,
    settings: Settings = Depends(get_settings_dep),
):
    """
    Scan a directory for PDF CVs and trigger bulk processing.
     Supports local directories and Google Drive folder links.

    Args:
        directory_path: Path to the directory (relative or absolute)
        drive_url: Optional Google Drive Folder URL to download from

    Returns:
        Summary of started tasks.
    """
    import glob
    import gdown
    import shutil
    
    target_dir = request.directory_path
    
    # Handle Google Drive URL
    if request.drive_url:
        try:
            # Create a download directory
            download_base = os.path.join(settings.upload_dir, "gdrive_downloads")
            os.makedirs(download_base, exist_ok=True)
            
            # Extract folder ID or use a hash of the URL to keep it unique but consistent
            # Simple approach: Use a timestamp or UUID for new download session to avoid conflicts?
            # Or better: just download to a temp folder
            session_dl_dir = os.path.join(download_base, str(uuid.uuid4())[:8])
            os.makedirs(session_dl_dir, exist_ok=True)
            
            logger.info(f"Downloading from Drive URL to {session_dl_dir}...")
            
            # Download folder (quiet mode to reduce log spam)
            gdown.download_folder(url=request.drive_url, output=session_dl_dir, quiet=False, remaining_ok=True)
            
            target_dir = session_dl_dir
            logger.info(f"Download complete: {target_dir}")
            
        except Exception as e:
             logger.error(f"Google Drive download failed: {e}")
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download from Drive: {str(e)}"
            )

    # Security check: Ensure generated path is safe if needed
    # For now, we allow any path as per user request "public shared folder"
    
    if not os.path.exists(target_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {target_dir}"
        )
        
    if not os.path.isdir(target_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {target_dir}"
        )
    
    # Scan for PDFs (recursively if it's a downloaded folder structure)
    # Using glob with recursive=True
    if request.drive_url:
         pdf_files = glob.glob(os.path.join(target_dir, "**", "*.pdf"), recursive=True)
    else:
         pdf_files = glob.glob(os.path.join(target_dir, "*.pdf"))
    
    triggered_count = 0
    errors = []
    
    for pdf_path in pdf_files:
        try:
            filename = os.path.basename(pdf_path)
            
            # Queue task directly with the file path
            # Note: The worker needs access to this path. 
            # If worker is in Docker, path must be mounted.
            
            process_cv_task.delay(
                file_path=os.path.abspath(pdf_path),
                original_filename=filename,
            )
            triggered_count += 1
            
        except Exception as e:
            logger.error(f"Failed to queue task for {pdf_path}: {e}")
            errors.append(f"{filename}: {e}")
            
    return {
        "message": f"Batch processing started for {triggered_count} files",
        "directory": target_dir,
        "found_files": len(pdf_files),
        "triggered_tasks": triggered_count,
        "errors": errors
    }
