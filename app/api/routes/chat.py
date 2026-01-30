"""
Chat API Routes.

Endpoints for:
- Streaming chat with RAG
- Managing conversation history
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_async_db
from app.schemas.chat import (
    ChatRequest,
    ChatHistoryResponse,
    CandidateCard,
    RetrievedChunk,
)
from app.services.chat.memory import get_conversation_memory
from app.services.chat.rag_chain import get_rag_chain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat Assistant"])


async def generate_sse_stream(
    session_id: str,
    message: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream for chat response.
    
    SSE Format:
    - data: {"type": "token", "content": "..."}
    - data: {"type": "candidates", "data": [...]}
    - data: {"type": "chunks", "data": [...]}  # Debug: retrieved CV chunks
    - data: {"type": "done"}
    """
    rag_chain = get_rag_chain()
    
    # Signal that we're starting
    yield f"data: {json.dumps({'type': 'start'})}\n\n"
    
    # Stream response
    candidates = []
    chunks = []
    try:
        async for event_type, content in rag_chain.chat(session_id, message, db):
            if event_type == "status":
                yield f"data: {json.dumps({'type': 'status', 'content': content}, ensure_ascii=False)}\n\n"
                continue
                
            event_data = {"type": "token", "content": content}
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        
        # Get candidates from the response
        candidates = await rag_chain.get_candidates_from_last_response(session_id)
        
        # Get retrieved chunks for debug
        chunks = rag_chain.get_retrieved_chunks(session_id)
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    # Send candidate cards if any
    if candidates:
        candidates_data = {
            "type": "candidates",
            "data": [c.model_dump() for c in candidates],
        }
        yield f"data: {json.dumps(candidates_data, ensure_ascii=False)}\n\n"
    
    # Send retrieved chunks for debug/transparency
    if chunks:
        chunks_data = {
            "type": "chunks",
            "data": [c.model_dump() for c in chunks],
        }
        yield f"data: {json.dumps(chunks_data, ensure_ascii=False)}\n\n"
    
    # Signal completion
    yield f"data: {json.dumps({'type': 'done'})}\n\n"



@router.post("")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Send a message and receive a streaming response.
    
    Uses Server-Sent Events (SSE) to stream tokens as they are generated.
    
    Event types:
    - `start`: Stream has started
    - `token`: A response token (partial text)
    - `candidates`: Array of candidate cards referenced in response
    - `done`: Stream has ended
    - `error`: An error occurred
    
    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/api/v1/chat');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            appendToOutput(data.content);
        }
    };
    ```
    """
    return StreamingResponse(
        generate_sse_stream(request.session_id, request.message, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    Get the conversation history for a session.
    
    Returns the last N messages (based on sliding window configuration).
    """
    memory = get_conversation_memory()
    messages = await memory.get_history(session_id)
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
    )


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear the conversation history for a session.
    
    This removes all messages and resets the conversation.
    """
    memory = get_conversation_memory()
    deleted = await memory.clear_session(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return {"message": f"Session {session_id} cleared successfully"}


@router.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """
    Get metadata about a chat session.
    
    Returns message count, TTL, and whether the session exists.
    """
    memory = get_conversation_memory()
    info = await memory.get_session_info(session_id)
    
    return info


@router.get("/candidates/{session_id}")
async def get_last_candidates(session_id: str):
    """
    Get candidate cards from the last assistant response.
    
    Useful for displaying candidate details after a search.
    """
    rag_chain = get_rag_chain()
    candidates = await rag_chain.get_candidates_from_last_response(session_id)
    
    return {
        "session_id": session_id,
        "candidates": [c.model_dump() for c in candidates],
        "count": len(candidates),
    }
