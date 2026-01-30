"""
Celery Background Tasks for CV Processing.

Heavy operations like OCR, LLM parsing, and embedding generation
run in background workers to avoid blocking API requests.
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from celery import Task

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class CVProcessingTask(Task):
    """Base task with error handling and progress updates."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")

    def update_progress(self, stage: str, progress: int, **extra):
        """Update task progress for status tracking."""
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": stage,
                "progress": progress,
                **extra,
            },
        )


@celery_app.task(bind=True, base=CVProcessingTask, name="cv.process")
def process_cv_task(
    self,
    file_path: str,
    original_filename: str,
) -> Dict[str, Any]:
    """
    Main CV processing pipeline.
    
    Stages:
    1. OCR & Text Extraction
    2. Layout Analysis (for two-column CVs)
    3. Vietnamese Preprocessing
    4. LLM Parsing
    5. Chunking
    6. Embedding Generation
    7. Database Storage
    """
    logger.info(f"Starting CV processing: {original_filename}")

    try:
        # Stage 1: OCR / Text Extraction
        self.update_progress("ocr_processing", 10, filename=original_filename)
        text = _extract_text(file_path)

        if not text or len(text.strip()) < 50:
            raise ValueError("Could not extract text from document")

        # Stage 2: Vietnamese Preprocessing
        self.update_progress("text_preprocessing", 25, filename=original_filename)
        processed_text = _preprocess_text(text)

        # Stage 3: LLM Parsing
        self.update_progress("llm_parsing", 40, filename=original_filename)
        resume = _parse_with_llm(processed_text, original_filename)

        # Stage 4: Calculate Experience
        self.update_progress("llm_parsing", 50, filename=original_filename)
        total_experience = _calculate_experience(resume)

        # Stage 4.5: Evaluate & Reformat CV Data Quality
        self.update_progress("evaluating", 55, filename=original_filename)
        resume, eval_result = _evaluate_cv_data(resume)
        logger.info(f"CV quality score: {eval_result.score:.1f}/10, issues: {len(eval_result.issues)}")

        # Stage 5: Chunking
        self.update_progress("chunking", 60, filename=original_filename)
        chunks = _create_chunks(processed_text, resume)

        # Stage 6: Enrichment
        self.update_progress("chunking", 70, filename=original_filename)
        enriched_chunks = _enrich_chunks(chunks, resume)

        # Stage 7: Embedding Generation
        self.update_progress("embedding", 80, filename=original_filename)
        chunks_with_embeddings = _generate_embeddings(enriched_chunks, resume)

        # Stage 8: Database Storage
        self.update_progress("embedding", 90, filename=original_filename)
        candidate_id = _save_to_database(
            resume,
            chunks_with_embeddings,
            total_experience,
            original_filename,
            processed_text,
        )

        # Stage 9: Update BM25 Index
        self.update_progress("indexing", 95, filename=original_filename)
        _update_bm25_index(candidate_id, chunks_with_embeddings)

        logger.info(f"CV processing complete: {original_filename} -> {candidate_id}")

        return {
            "candidate_id": candidate_id,
            "filename": original_filename,
            "validation_warnings": resume.validation_warnings,
        }

    except Exception as e:
        logger.error(f"CV processing failed: {e}")
        raise


def _extract_text(file_path: str) -> str:
    """Extract text from document (PDF/image)."""
    from app.services.ingestion.ocr import OCRService

    ext = os.path.splitext(file_path)[1].lower()

    ocr = OCRService()

    if ext == ".pdf":
        return ocr.extract_text_hybrid(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        return ocr.extract_text_only(file_path)
    elif ext in {".docx", ".doc"}:
        # Use python-docx for Word documents
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.warning("python-docx not installed")
            return ocr.extract_text_only(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _preprocess_text(text: str) -> str:
    """Apply Vietnamese preprocessing."""
    from app.services.ingestion.preprocessor import get_preprocessor

    preprocessor = get_preprocessor()
    return preprocessor.preprocess_for_embedding(text)


def _parse_with_llm(text: str, filename: str):
    """Parse CV text with LLM."""
    from app.services.parsing.llm_parser import get_parser

    parser = get_parser()
    return parser.parse_resume_with_fallback(text, filename)


def _calculate_experience(resume) -> float:
    """Calculate total experience using merge intervals."""
    from app.services.utils.experience import calculate_total_experience

    if not resume.work_experience:
        return 0.0

    intervals = [
        (exp.start_date, exp.end_date)
        for exp in resume.work_experience
        if exp.start_date
    ]

    return calculate_total_experience(intervals)


def _evaluate_cv_data(resume):
    """Evaluate and reformat CV data if needed."""
    from app.services.parsing.cv_evaluator import get_cv_evaluator
    
    evaluator = get_cv_evaluator()
    return evaluator.evaluate_and_reformat(resume)


def _create_chunks(text: str, resume):
    """Create section-aware chunks."""
    from app.services.parsing.chunker import SectionAwareChunker

    chunker = SectionAwareChunker()
    return chunker.chunk_document(text)


def _enrich_chunks(chunks, resume):
    """Enrich chunks with contextual metadata."""
    from app.services.parsing.enricher import ContextualEnricher

    enricher = ContextualEnricher()
    return enricher.enrich_chunks(chunks, resume)


def _generate_embeddings(chunks, resume) -> list:
    """Generate embeddings for chunks and summary."""
    from app.services.embedding.embedder import get_embedding_service
    from app.services.parsing.enricher import ContextualEnricher

    embedding_service = get_embedding_service()
    enricher = ContextualEnricher()

    results = []

    for chunk in chunks:
        # Get enriched content for embedding
        enriched = chunk.metadata.get("enriched_content", chunk.content)
        embedding = embedding_service.embed_document(enriched)

        results.append({
            "chunk": chunk,
            "embedding": embedding,
            "enriched_content": enriched,
        })

    # Generate summary embedding
    summary_context = enricher.build_summary_context(resume)
    summary_embedding = embedding_service.embed_document(summary_context)

    return {
        "chunks": results,
        "summary_embedding": summary_embedding,
    }


def _save_to_database(
    resume,
    embeddings_data: dict,
    total_experience: float,
    filename: str,
    raw_text: str,
) -> str:
    """Save candidate and chunks to database."""
    from app.models.candidate import Candidate, Chunk

    db = SessionLocal()

    try:
        # Create candidate
        candidate_id = str(uuid.uuid4())

        candidate = Candidate(
            id=candidate_id,
            full_name=resume.full_name,
            email=resume.email,
            phone=resume.phone,
            headline=resume.headline,
            summary=resume.summary,
            raw_resume=resume.model_dump(mode='json'),
            summary_embedding=embeddings_data["summary_embedding"],
            total_experience_years=total_experience,
            top_skills=resume.get_all_skills()[:20],
            validation_warnings=resume.validation_warnings,
            source_filename=filename,
            raw_text=raw_text,
        )

        db.add(candidate)

        # Create chunks
        for item in embeddings_data["chunks"]:
            chunk_obj = item["chunk"]

            db_chunk = Chunk(
                id=chunk_obj.id,
                candidate_id=candidate_id,
                parent_id=chunk_obj.parent_id,
                section=chunk_obj.section.value,
                subsection=chunk_obj.subsection,
                content=chunk_obj.content,
                enriched_content=item["enriched_content"],
                embedding=item["embedding"],
                chunk_metadata=chunk_obj.metadata,
                order_index=chunk_obj.order_index,
            )

            db.add(db_chunk)

        db.commit()
        logger.info(f"Saved candidate {candidate_id} with {len(embeddings_data['chunks'])} chunks")

        return candidate_id

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def _update_bm25_index(candidate_id: str, embeddings_data: dict) -> None:
    """
    Update BM25 index with chunks from newly saved candidate.
    
    Args:
        candidate_id: The saved candidate's ID
        embeddings_data: Dict containing chunks with embeddings
    """
    try:
        from app.services.search.hybrid import get_search_engine
        
        search_engine = get_search_engine()
        
        chunk_ids = [item["chunk"].id for item in embeddings_data["chunks"]]
        contents = [item["enriched_content"] for item in embeddings_data["chunks"]]
        
        search_engine.add_candidate_chunks(
            chunk_ids=chunk_ids,
            candidate_id=candidate_id,
            contents=contents,
        )
        
        logger.info(f"Updated BM25 index with {len(chunk_ids)} chunks for candidate {candidate_id}")
        
    except Exception as e:
        # Log error but don't fail the task - BM25 index can be rebuilt at startup
        logger.error(f"Failed to update BM25 index for candidate {candidate_id}: {e}")


@celery_app.task(bind=True, base=CVProcessingTask, name="cv.generate_embeddings")
def generate_embeddings_task(
    self,
    candidate_id: str,
) -> Dict[str, Any]:
    """
    Regenerate embeddings for an existing candidate.
    
    Useful when switching embedding models.
    """
    from app.models.candidate import Candidate, Chunk
    from app.services.embedding.embedder import get_embedding_service

    db = SessionLocal()

    try:
        # Get candidate
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if not candidate:
            raise ValueError(f"Candidate not found: {candidate_id}")

        embedding_service = get_embedding_service()

        # Regenerate summary embedding
        self.update_progress("embedding", 30)
        if candidate.summary:
            candidate.summary_embedding = embedding_service.embed_document(candidate.summary)

        # Regenerate chunk embeddings
        chunks = db.query(Chunk).filter(Chunk.candidate_id == candidate_id).all()

        for i, chunk in enumerate(chunks):
            progress = 30 + int(70 * (i + 1) / len(chunks))
            self.update_progress("embedding", progress)

            text = chunk.enriched_content or chunk.content
            chunk.embedding = embedding_service.embed_document(text)

        db.commit()

        return {
            "candidate_id": candidate_id,
            "chunks_updated": len(chunks),
        }

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
