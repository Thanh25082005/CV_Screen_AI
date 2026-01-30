
import os
import asyncio
import json
from app.services.chat.query_transformer import QueryTransformer

# Mock Groq client to avoid actual API calls if needed, 
# but for this verification we want to see actual LLM output if possible.
# Assuming environment variables are set.

async def main():
    transformer = QueryTransformer()
    
    test_cases = [
        "Tìm dev Python ở HN có 3 năm kinh nghiệm",
        "Có ứng viên nào biết về React và Nodejs không?",
        "Tóm tắt hồ sơ của Nguyễn Văn A",
        "Chào bạn, bạn khỏe không?",
    ]
    
    print("--- Starting Query Transformation Verification in ---")
    
    for msg in test_cases:
        print(f"\nUser Message: {msg}")
        try:
            result = await transformer.transform_async(msg)
            print("Transformed Result:")
            print(json.dumps({
                "semantic_query": result.semantic_query,
                "keyword_string": result.keyword_string,
                "filters": result.filters,
                "intent": result.intent,
                "is_search_needed": result.is_search_needed,
                "explanation": result.explanation
            }, indent=2, ensure_ascii=False))
            
            # Simple assertions
            if "Tìm dev Python" in msg:
                assert result.intent == "search"
                assert "Python" in result.keyword_string
                assert result.filters.get("location") == "HN" or result.filters.get("location") == "Hanoi"
                
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
