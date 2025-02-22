# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

from typing import Dict, Any, List
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text

from ..models.query import SQLQuery
from ..models.response import QueryResult

class SQLExecutor:
    def __init__(self, connection_string: str):
        self.engine: AsyncEngine = create_async_engine(
            connection_string,
            echo=False,
            pool_pre_ping=True
        )

    async def execute(self, query: SQLQuery) -> QueryResult:
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    text(query.query),
                    query.parameters or {}
                )
                
                # Convert results to dictionaries more safely
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result]
                
                return QueryResult(
                    source="sql",
                    data=rows,
                    metadata={"row_count": len(rows)}
                )
                
        except Exception as e:
            return QueryResult(
                source="sql",
                data=[],
                metadata={"error": str(e)}
            ) 