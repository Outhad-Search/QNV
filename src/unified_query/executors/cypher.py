# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

from typing import Dict, Any, List
from neo4j import AsyncGraphDatabase, AsyncDriver
from ..models.query import CypherQuery
from ..models.response import QueryResult

class CypherExecutor:
    def __init__(self, uri: str, username: str, password: str):
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(username, password)
        )

    async def execute(self, query: CypherQuery) -> QueryResult:
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query.query,
                    query.parameters or {}
                )
                records = await result.data()
                
                return QueryResult(
                    source="cypher",
                    data=records,
                    metadata={"record_count": len(records)}
                )
                
        except Exception as e:
            return QueryResult(
                source="cypher",
                data=[],
                metadata={"error": str(e)}
            )

    async def close(self):
        await self.driver.close() 