"""
RAG Chain for Recruiter Assistant ChatBot.

Orchestrates the full RAG pipeline:
1. Get chat history from memory
2. Transform query (extract search query + filters)
3. Search candidates if needed
4. Build context and generate response with streaming
"""

import json
import logging
from typing import Optional, List, AsyncGenerator, Dict, Any

from groq import Groq

from app.config import get_settings
from app.schemas.chat import (
    ChatMessage,
    MessageRole,
    CandidateCard,
    TransformedQuery,
    RetrievedChunk,
)
from app.schemas.search import SearchRequest, SearchType
from app.services.chat.memory import ConversationMemory, get_conversation_memory
from app.services.chat.query_transformer import QueryTransformer, get_query_transformer
from app.services.search.hybrid import HybridSearchEngine, get_search_engine

settings = get_settings()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """# ROLE
Bạn là trợ lý tuyển dụng thông minh. Bạn CHỈ được phép trả lời các câu hỏi liên quan đến ứng viên trong hệ thống.

# QUY TẮC BẮT BUỘC (CRITICAL)

## 1. CHỈ TRẢ LỜI DỰA TRÊN DATABASE
- Bạn CHỈ được phép trả lời dựa HOÀN TOÀN vào dữ liệu JSON trong phần [CONTEXT]
- Nếu KHÔNG CÓ dữ liệu trong [CONTEXT], bạn PHẢI nói: "Không tìm thấy dữ liệu phù hợp trong hệ thống."
- TUYỆT ĐỐI KHÔNG được bịa đặt, phỏng đoán, hoặc thêm thông tin không có trong database

## 2. TRUNG THỰC VÀ CHÍNH XÁC
- Nếu một trường là "null" hoặc không có: Nói rõ "Thông tin này không có trong hồ sơ"
- Nếu không tìm thấy ứng viên: Nói rõ "Không tìm thấy ứng viên nào phù hợp với yêu cầu"
- Nếu câu hỏi nằm ngoài phạm vi tuyển dụng/CV: Từ chối lịch sự

## 3. PHẠM VI ĐƯỢC PHÉP TRẢ LỜI
✅ Được phép:
- Thông tin ứng viên (tên, email, số điện thoại, kỹ năng, kinh nghiệm)
- Tìm kiếm ứng viên theo tiêu chí (skills, location, experience)
- So sánh ứng viên dựa trên dữ liệu thực
- Đếm số lượng ứng viên trong database
- Thông tin về projects, education, certifications của ứng viên

❌ KHÔNG được phép:
- Trả lời câu hỏi không liên quan đến tuyển dụng/CV
- Đưa ra nhận xét chủ quan không dựa trên dữ liệu
- Phỏng đoán về khả năng, tính cách của ứng viên
- Trả lời về các chủ đề: chính trị, tôn giáo, giải trí, tin tức...

## 4. ĐỊNH DẠNG MARKDOWN
- Dùng `###` cho tên ứng viên (e.g., `### 👤 VU VAN THANH`)
- Dùng bảng Markdown cho so sánh hoặc thông tin có cấu trúc
- Dùng danh sách `-` cho skills
- PHẢI có dòng trống trước và sau heading, table, list

# VÍ DỤ PHẢN HỒI ĐÚNG

**Khi so sánh ứng viên:**
```
Dưới đây là bảng so sánh giữa Nguyen Van A và Tran Van B:

| Tiêu chí | 👤 NGUYEN VAN A | 👤 TRAN VAN B |
| :--- | :--- | :--- |
| **Kinh nghiệm** | 5 năm (Senior) | 3 năm (Mid-level) |
| **Kỹ năng chính** | Python, DevOps, AWS | Python, Django, React |
| **Điểm mạnh** | Có chứng chỉ AWS, kinh nghiệm dồi dào | Fullstack, tiếng Anh tốt |

**Kết luận:**
- Nếu cần vị trí thiên về hạ tầng/backend sâu: Chọn **Nguyen Van A**.
- Nếu cần làm sản phẩm nhanh (Fullstack): Chọn **Tran Van B**.
```

**Khi tìm thấy ứng viên:**
```
### 👤 NGUYEN VAN A

| Thông tin | Chi tiết |
| :--- | :--- |
| **Email** | example@email.com |
| **Kinh nghiệm** | 5.2 năm |

**Kỹ năng:** Python, FastAPI, Docker
```

**Khi KHÔNG tìm thấy:**
"Xin lỗi, tôi không tìm thấy ứng viên nào phù hợp với tiêu chí 'Data Scientist tại Đà Nẵng' trong hệ thống. Hiện tại database có 2 ứng viên."

**Khi câu hỏi ngoài phạm vi:**
"Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến tuyển dụng và thông tin ứng viên trong hệ thống. Bạn có thể hỏi tôi: 'Tìm ứng viên Python', 'Ai có kinh nghiệm React?', 'Thông tin chi tiết về ứng viên X'."

# EXECUTION
Luôn đọc kỹ [CONTEXT], chỉ trả lời dựa trên dữ liệu thực, và từ chối lịch sự nếu không có thông tin hoặc câu hỏi ngoài phạm vi."""

CONTEXT_TEMPLATE = """
[CONTEXT] 
{candidate_context}
"""

PARSING_RECOVERY_PROMPT = """
# PARSING ERROR RECOVERY MODE

⚠️ The user has indicated that the requested data DOES exist in the profile, but the previous extraction failed. You must perform a deep re-read of the raw text.

## STRICT INSTRUCTIONS:
1. **DO NOT use default answers** like "Not mentioned", "Không rõ", "N/A" etc.
2. **Re-read the ENTIRE raw text** of each candidate carefully, line by line.
3. If you cannot find the candidate's name, use "Ứng viên [Số thứ tự]" (e.g., "Ứng viên 1", "Ứng viên 2").
4. **Search for keywords** in the raw text such as: Python, Kinh nghiệm, Nơi làm việc, Học vấn, Kỹ năng, etc.
5. **Extract content greedily**: Even if the format doesn't match a perfect table structure, extract any text near the relevant keywords.
6. If you find partial information (e.g., "3 năm làm việc tại công ty ABC"), format it as best you can.

## EXAMPLE OUTPUT FOR INCOMPLETE DATA:
```
### 👤 Ứng viên 1

| Field | Description |
| :--- | :--- |
| **Tên** | (Trích từ văn bản: "Nguyễn Văn A") |
| **Kinh nghiệm** | ~3 năm (Trích: "đã làm việc 3 năm tại Công ty XYZ") |
| **Kỹ năng Python** | ✅ Có đề cập (Trích: "sử dụng Python trong dự án AI") |

**Ghi chú:** Hồ sơ này có định dạng không chuẩn, thông tin trên được trích xuất thủ công từ văn bản thô.
```

Now, re-process the [CONTEXT] data using these recovery rules."""



class RAGChain:
    """
    Main RAG orchestration chain for the ChatBot.
    
    Combines:
    - Conversation memory (Redis)
    - Query transformation (LLM)
    - Hybrid search (BM25 + Vector)
    - Response generation with streaming
    """
    
    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        transformer: Optional[QueryTransformer] = None,
        search_engine: Optional[HybridSearchEngine] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize RAG chain with dependencies."""
        self.memory = memory or get_conversation_memory()
        self.transformer = transformer or get_query_transformer()
        self.search_engine = search_engine or get_search_engine()
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.chat_model
        self._client: Optional[Groq] = None
        # Track retrieved chunks per session for debug output
        self._last_retrieved_chunks: Dict[str, List[RetrievedChunk]] = {}

    
    def _get_client(self) -> Groq:
        """Get or create Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    async def chat(
        self,
        session_id: str,
        message: str,
        db_session,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Process a chat message and stream the response.
        
        Args:
            session_id: Unique session identifier
            message: User message
            db_session: Database session for search
            
        Yields:
            (type, content) tuples where type is 'token' or 'status'
        """
        # Step 1: Get conversation history
        history = await self.memory.get_history(session_id)
        logger.info(f"Session {session_id}: Got {len(history)} messages from history")
        
        # Step 2: Save user message to history
        await self.memory.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=message,
        )
        
        # Step 3: Transform query
        yield ("status", "Đang phân tích yêu cầu...")
        transformed = await self.transformer.transform_async(message, history)
        logger.info(f"Transformed query: {transformed.search_query}, intent: {transformed.intent}")
        
        # Step 3.5: Handle off-topic questions
        if transformed.intent == "off_topic":
            off_topic_response = (
                "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến **tuyển dụng và thông tin ứng viên** trong hệ thống.\n\n"
                "Bạn có thể hỏi tôi những câu như:\n"
                "- 🔍 \"Tìm ứng viên Python Developer\"\n"
                "- 📋 \"Ai có kinh nghiệm React?\"\n"
                "- 👤 \"Thông tin chi tiết về ứng viên Vu Van Thanh\"\n"
                "- 📊 \"So sánh 2 ứng viên có kỹ năng Machine Learning\"\n"
                "- 🏢 \"Có ứng viên nào ở Hà Nội không?\""
            )
            yield ("token", off_topic_response)
            
            # Save to history
            await self.memory.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=off_topic_response,
            )
            return
        
        # Step 3.6: Handle list_all intent - fetch all candidates from DB
        candidates: List[CandidateCard] = []
        candidate_context = ""
        
        if transformed.intent == "list_all":
            yield ("status", "Đang tải danh sách ứng viên...")
            
            from sqlalchemy import select
            from app.models.candidate import Candidate
            
            # Fetch all candidates
            result = await db_session.execute(
                select(Candidate).order_by(Candidate.created_at.desc()).limit(20)
            )
            all_candidates = result.scalars().all()
            
            if all_candidates:
                # Build candidate cards
                candidates = []
                json_context_data = []
                
                for cand in all_candidates:
                    card = CandidateCard(
                        candidate_id=cand.id,
                        full_name=cand.full_name,
                        headline=cand.headline,
                        total_experience_years=cand.total_experience_years or 0,
                        top_skills=cand.top_skills[:5] if cand.top_skills else [],
                        email=cand.email,
                        # phone=cand.phone,  # Schema does not have phone
                    )
                    candidates.append(card)
                    
                    # Build context for LLM
                    json_context_data.append({
                        "name": cand.full_name,
                        "email": cand.email,
                        "phone": cand.phone,
                        "headline": cand.headline,
                        "experience_years": cand.total_experience_years,
                        "skills": cand.top_skills[:10] if cand.top_skills else [],
                        "summary": cand.summary,
                    })
                
                import json
                candidate_context = CONTEXT_TEMPLATE.format(
                    candidate_context=json.dumps(json_context_data, ensure_ascii=False, indent=2)
                )
                candidate_context = f"[DATABASE INFO] Tổng số ứng viên: {len(all_candidates)}. Dưới đây là danh sách:\n\n" + candidate_context
                
                logger.info(f"List all: Found {len(all_candidates)} candidates")
            else:
                candidate_context = "[DATABASE INFO] Hiện tại chưa có ứng viên nào trong hệ thống."
        
        # Step 4: Search candidates if needed (for specific search queries)
        elif transformed.is_search_needed and (transformed.semantic_query or transformed.keyword_string):
            yield ("status", "Đang tìm kiếm và đánh giá hồ sơ...")
            # Build search request with filters
            search_request = SearchRequest(
                query=transformed.semantic_query, # Use semantic query for vector
                keyword_query=transformed.keyword_string, # Use keyword string for BM25
                search_type=SearchType.HYBRID,
                expand_query=True,
                top_k=settings.chat_max_candidates,
                min_experience_years=transformed.filters.get("min_experience_years"),
                required_skills=transformed.filters.get("required_skills", []),
                location=transformed.filters.get("location"),
            )
            
            try:
                # Execute 3-Layer Search Strategy
                candidates, retrieved_chunks, search_note = await self._search_with_fallback(
                    search_request=search_request,
                    db_session=db_session
                )
                
                # Build JSON context
                json_context_data = []
                for i, card in enumerate(candidates[:settings.chat_max_candidates], 1):
                    # Find chunks for this candidate
                    matches = [c for c in retrieved_chunks if c.candidate_name == card.full_name]
                    
                    candidate_data = {
                        "full_name": card.full_name,
                        "headline": card.headline,
                        "total_experience_years": card.total_experience_years,
                        "top_skills": card.top_skills,
                        "email": card.email,
                        "match_score": card.match_score,
                        "retrieved_sections": [
                            {
                                "section": m.section,
                                "content": m.content,
                                "score": m.score
                            }
                            for m in matches
                        ]
                    }
                    json_context_data.append(candidate_data)

                self._last_retrieved_chunks[session_id] = retrieved_chunks
                
                if json_context_data:
                    import json
                    candidate_context_json = json.dumps(json_context_data, ensure_ascii=False, indent=2)
                    
                    candidate_context = CONTEXT_TEMPLATE.format(
                        candidate_context=candidate_context_json
                    )
                    # Add search note to context (if relaxed search was used)
                    if search_note:
                         candidate_context = f"LƯU Ý HỆ THỐNG: {search_note}\n\n" + candidate_context
                else:
                    # Get total count for context
                    from sqlalchemy import func, select
                    from app.models.candidate import Candidate
                    count_result = await db_session.execute(select(func.count(Candidate.id)))
                    total_count = count_result.scalar() or 0
                    candidate_context = f"[DATABASE INFO] Không tìm thấy ứng viên phù hợp với tiêu chí. Tổng số ứng viên trong hệ thống: {total_count}."
                    
                logger.info(f"Found {len(candidates)} candidates via strategy.")

                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                candidate_context = f"Lỗi khi tìm kiếm: {str(e)}"
        
        # Step 5: Build messages for LLM
        # Check if user is requesting parsing recovery (frustrated with "not mentioned" answers)
        recovery_keywords = [
            "lỗi", "parsing error", "bị lỗi", "không đúng", "đã có trong hồ sơ",
            "dữ liệu có tồn tại", "thử lại", "retry", "trích xuất lại",
            "không tìm thấy", "sai rồi", "lười"
        ]
        is_recovery_request = any(kw in message.lower() for kw in recovery_keywords)
        
        if is_recovery_request:
            logger.info(f"Detected parsing recovery request from user")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": PARSING_RECOVERY_PROMPT},
            ]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add context if we have candidates
        if candidate_context:
            messages.append({"role": "system", "content": candidate_context})
        
        # Add conversation history (last few messages)
        for msg in history[-6:]:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # Step 6: Generate response with quality evaluation loop
        yield ("status", "Đang tổng hợp câu trả lời...")
        client = self._get_client()
        
        # Import critic for response evaluation
        from app.services.chat.response_critic import get_response_critic
        critic = get_response_critic()
        
        full_response = ""
        best_response = ""
        best_score = 0.0
        max_attempts = 3
        
        for attempt in range(max_attempts):
            current_response = ""
            
            try:
                # Build messages for this attempt
                attempt_messages = messages.copy()
                
                # If retry, add critic feedback
                if attempt > 0 and hasattr(self, '_last_critic_result'):
                    feedback_prompt = critic.get_regeneration_prompt(self._last_critic_result)
                    attempt_messages.insert(1, {"role": "system", "content": feedback_prompt})
                    yield ("status", f"Đang cải thiện câu trả lời (lần {attempt + 1})...")
                
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=attempt_messages,
                    temperature=0.7 if attempt == 0 else 0.5,  # Lower temp on retry
                    max_tokens=1000,
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        current_response += token
                        # Buffer response, do not stream yet
                
                # Evaluate response quality
                if attempt < max_attempts - 1:
                    yield ("status", "Đang đánh giá chất lượng câu trả lời...")
                    critic_result = critic.evaluate(message, current_response, candidate_context)
                    
                    logger.info(f"Response critic score (attempt {attempt + 1}): {critic_result.score:.1f}/10")
                    
                    # Keep track of best response
                    if critic_result.score > best_score:
                        best_score = critic_result.score
                        best_response = current_response
                    
                    # Check if good enough
                    if not critic.should_retry(critic_result, attempt):
                        full_response = current_response
                        break
                    
                    # Store for next iteration
                    self._last_critic_result = critic_result
                else:
                    # Final attempt - use this response
                    full_response = current_response
                    
            except Exception as e:
                logger.error(f"LLM generation failed (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                    full_response = error_msg
                continue
        
        # Use best response if current is worse
        if best_score > 0 and not full_response:
            full_response = best_response
        
        # Yield the final response immediately for maximum speed
        # User requested to remove streaming effect for better performance
        chunk_size = 1024  # Large chunk size
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i + chunk_size]
            yield ("token", chunk)
            # No sleep -> Instant return
        
        # Step 7: Save assistant response to history
        await self.memory.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            candidates=candidates,
        )
        
        logger.info(f"Session {session_id}: Completed response ({len(full_response)} chars)")
    
    async def get_candidates_from_last_response(
        self,
        session_id: str,
    ) -> List[CandidateCard]:
        """Get candidate cards from the last assistant message."""
        history = await self.memory.get_history(session_id, limit=1)
        
        for msg in reversed(history):
            if msg.role == MessageRole.ASSISTANT and msg.candidates:
                return msg.candidates
        
        return []

    async def _search_with_fallback(
        self,
        search_request: SearchRequest,
        db_session
    ) -> tuple[List[CandidateCard], List[RetrievedChunk], str]:
        """
        Execute search with 3-Layer Fallback Strategy.
        
        Returns:
            (candidates, debug_chunks, search_note)
        """
        candidates = []
        retrieved_chunks = []
        search_note = ""

        # === LAYER 1: STRICT SEARCH ===
        logger.info("Layer 1: Strict Search")
        response = await self.search_engine.search(search_request, db_session)
        
        if response.results:
            logger.info(f"Layer 1 success: Found {len(response.results)} candidates")
            return self._process_search_results(response) + ("",)

        # === LAYER 2: RELAXED SEARCH (Soft Filters) ===
        # Create relaxed request: Remove location, reduce experience
        logger.info("Layer 1 empty. Trying Layer 2: Relaxed Search")
        
        relaxed_request = SearchRequest(
            query=search_request.query,
            search_type=SearchType.HYBRID,
            expand_query=True,
            top_k=search_request.top_k,
            # Remove location filter
            location=None, 
            # Reduce experience requirement by 30% if exists, or remove if < 1 year
            min_experience_years=max(0.0, search_request.min_experience_years * 0.7) if search_request.min_experience_years else None,
            required_skills=search_request.required_skills, # Keep skills strictly for now
        )
        
        response = await self.search_engine.search(relaxed_request, db_session)
        if response.results:
            search_note = "Không tìm thấy ứng viên khớp 100% tiêu chí (Layer 1). Đây là các ứng viên GẦN ĐÚNG NHẤT (đã nới lỏng tiêu chí Location/Experience)."
            logger.info(f"Layer 2 success: Found {len(response.results)} candidates")
            return self._process_search_results(response) + (search_note,)

        # === LAYER 3: SEMANTIC FALLBACK (Vector Only) ===
        # Remove all filters, just semantic search
        logger.info("Layer 2 empty. Trying Layer 3: Semantic Fallback")
        
        fallback_request = SearchRequest(
            query=search_request.query,
            search_type=SearchType.SEMANTIC, # Pure vector search
            expand_query=True,
            top_k=3, # Limit to top 3 for fallback
            location=None,
            min_experience_years=None,
            required_skills=[], # Remove skill filter too
        )
        
        response = await self.search_engine.search(fallback_request, db_session)
        if response.results:
            search_note = "Không tìm thấy ứng viên phù hợp tiêu chí lọc. Đây là những ứng viên có nội dung hồ sơ LIÊN QUAN NHẤT theo ý nghĩa (Semantic Match)."
            logger.info(f"Layer 3 success: Found {len(response.results)} candidates")
            return self._process_search_results(response) + (search_note,)

        return [], [], "Không tìm thấy dữ liệu nào, kể cả khi tra cứu ngữ nghĩa."

    def _process_search_results(self, response) -> tuple[List[CandidateCard], List[RetrievedChunk]]:
        """Convert SearchResponse to CandidateCards and RetrievedChunks."""
        candidates = []
        retrieved_chunks = []
        
        for result in response.results:
            # Create candidate card
            card = CandidateCard(
                candidate_id=result.candidate_id,
                full_name=result.full_name or "Unknown",
                headline=result.headline,
                email=result.email,
                total_experience_years=result.total_experience_years,
                top_skills=result.top_skills[:5] if result.top_skills else [],
                match_score=result.combined_score,
            )
            candidates.append(card)
            
            # Store chunks
            for chunk in result.matched_chunks[:3]:
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        candidate_name=card.full_name,
                        section=chunk.section,
                        content=chunk.content[:500],
                        score=chunk.score,
                        match_type=chunk.match_type,
                    )
                )
        return candidates, retrieved_chunks

    def get_retrieved_chunks(self, session_id: str) -> List[RetrievedChunk]:
        """
        Get the retrieved chunks from the last search for debugging.
        
        Returns:
            List of RetrievedChunk with content from CV sections used in response.
        """
        return self._last_retrieved_chunks.get(session_id, [])



# Singleton instance
_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get or create the RAG chain singleton."""
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
