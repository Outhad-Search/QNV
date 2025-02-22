# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging
from unified_query.config import Settings
from unified_query.models.query import UnifiedQuery, SQLQuery, CypherQuery, VectorQuery
from unified_query.orchestrator import QueryOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_orchestrator() -> AsyncGenerator[QueryOrchestrator, None]:
    """Context manager for creating and cleaning up the orchestrator"""
    # Load settings from environment variables or .env file
    settings = Settings()
    
    orchestrator = QueryOrchestrator(
        sql_connection=settings.SQL_CONNECTION_STRING,
        neo4j_uri=settings.NEO4J_URI,
        neo4j_auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        vector_dimension=settings.VECTOR_DIMENSION
    )
    
    try:
        yield orchestrator
    finally:
        await orchestrator.close()

async def execute_query_with_retry(
    orchestrator: QueryOrchestrator,
    query: UnifiedQuery,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> dict:
    """Execute a query with retry logic"""
    for attempt in range(max_retries):
        try:
            response = await orchestrator.execute(query)
            
            if response.errors:
                logger.warning(f"Query completed with errors: {response.errors}")
            
            return {
                "success": True,
                "data": response.results,
                "execution_time": response.execution_time,
                "errors": response.errors
            }
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "data": None
                }

async def main():
    async with get_orchestrator() as orchestrator:
        # Example of a complex query combining all three systems
        query = UnifiedQuery(
            sql=SQLQuery(
                query="""
                WITH user_interests AS (
                    SELECT user_id, array_agg(interest) as interests
                    FROM user_interests
                    GROUP BY user_id
                )
                SELECT u.*, ui.interests
                FROM users u
                LEFT JOIN user_interests ui ON u.id = ui.user_id
                WHERE u.active = :active
                """,
                parameters={"active": True}
            ),
            cypher=CypherQuery(
                query="""
                MATCH (u:User)-[r:PURCHASED]->(p:Product)
                WHERE p.category IN $categories
                WITH u, COUNT(r) as purchase_count
                WHERE purchase_count > $min_purchases
                RETURN u.id, purchase_count
                ORDER BY purchase_count DESC
                LIMIT 10
                """,
                parameters={
                    "categories": ["Electronics", "Books"],
                    "min_purchases": 5
                }
            ),
            vector=VectorQuery(
                vector=[0.1] * 128,
                k=5,
                filter_params={"category": "Electronics"}
            )
        )

        try:
            # Execute query with retry logic
            result = await execute_query_with_retry(orchestrator, query)
            
            if result["success"]:
                print("\nQuery Results:")
                for source_result in result["data"]:
                    print(f"\nSource: {source_result.source}")
                    print(f"Records: {len(source_result.data)}")
                    print("Sample data:", source_result.data[:2])
                    print("Metadata:", source_result.metadata)
                
                print(f"\nTotal execution time: {result['execution_time']:.2f} seconds")
            else:
                print(f"Query failed: {result['error']}")

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(main()) 