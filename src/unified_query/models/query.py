# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class SQLQuery(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
class CypherQuery(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VectorQuery(BaseModel):
    vector: List[float]
    k: int = 10
    filter_params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UnifiedQuery(BaseModel):
    sql: Optional[SQLQuery] = None
    cypher: Optional[CypherQuery] = None
    vector: Optional[VectorQuery] = None
    combine_results: bool = True 