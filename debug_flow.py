
import asyncio
import os
import sys

# Add app to path
sys.path.append(os.getcwd())

from app.services.chat.query_transformer import QueryTransformer
from app.services.search.hybrid import HybridSearchEngine
from app.schemas.search import SearchRequest, SearchType
from app.core.database import AsyncSessionLocal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug():
    print("🚀 Starting Debug Flow for: 'ai có kỹ năng về python'")
    
    # 1. Test Query Transformer
    transformer = QueryTransformer()
    query = "ai có kỹ năng về python"
    
    print("\n--- 1. Query Transformation ---")
    transformed = await transformer.transform_async(query, history=[])
    print(f"Original: {query}")
    print(f"Transformed: {transformed.dict()}")
    
    # 2. Test Search (Strict)
    print("\n--- 2. Header Search Testing ---")
    
    # Simulate what RAG chain does
    filters = transformed.filters
    search_query = transformed.search_query
    
    # Construct SearchRequest
    req = SearchRequest(
        query=search_query,
        search_type=SearchType.HYBRID,
        expand_query=True,
        top_k=5,
        location=filters.get("location"),
        min_experience_years=filters.get("min_experience_years"),
        required_skills=filters.get("required_skills", []),
    )
    
    print(f"Search Request: {req}")

    async with AsyncSessionLocal() as session:
        engine = HybridSearchEngine()
        
        # Test Strict Search
        print("\n--- 3. Executing Strict Search ---")
        try:
            results = await engine.search(req, session)
            print(f"Found {len(results.results)} results")
            for r in results.results:
                print(f" - {r.full_name} (Score: {r.combined_score})")
                print(f"   Skills: {r.top_skills}")
        except Exception as e:
            print(f"Error in search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())
