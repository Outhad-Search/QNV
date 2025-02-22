from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class QueryResult(BaseModel):
    source: str  # 'sql', 'cypher', or 'vector'
    data: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UnifiedResponse(BaseModel):
    results: List[QueryResult]
    execution_time: float
    errors: Optional[List[str]] = None 