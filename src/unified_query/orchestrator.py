# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

import asyncio
import time
from typing import Optional, List
from .models.query import UnifiedQuery
from .models.response import UnifiedResponse, QueryResult
from .executors.sql import SQLExecutor
from .executors.cypher import CypherExecutor
from .executors.vector import VectorExecutor

class QueryOrchestrator:
    def __init__(
        self,
        sql_connection: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[tuple] = None,
        vector_dimension: Optional[int] = None
    ):
        self.sql_executor = SQLExecutor(sql_connection) if sql_connection else None
        self.cypher_executor = (
            CypherExecutor(neo4j_uri, neo4j_auth[0], neo4j_auth[1])
            if neo4j_uri and neo4j_auth else None
        )
        self.vector_executor = (
            VectorExecutor(vector_dimension)
            if vector_dimension else None
        )

    async def execute(self, query: UnifiedQuery) -> UnifiedResponse:
        start_time = time.time()
        tasks = []
        errors = []

        # Create tasks for each query type
        if query.sql and self.sql_executor:
            tasks.append(self.sql_executor.execute(query.sql))
        if query.cypher and self.cypher_executor:
            tasks.append(self.cypher_executor.execute(query.cypher))
        if query.vector and self.vector_executor:
            tasks.append(self.vector_executor.execute(query.vector))

        # Execute all queries concurrently
        results: List[QueryResult] = []
        if tasks:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in completed_tasks:
                if isinstance(result, Exception):
                    errors.append(str(result))
                else:
                    results.append(result)

        execution_time = time.time() - start_time

        return UnifiedResponse(
            results=results,
            execution_time=execution_time,
            errors=errors if errors else None
        )

    async def close(self):
        if self.cypher_executor:
            await self.cypher_executor.close() 