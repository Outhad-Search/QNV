# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

from typing import List, Dict, Any
import numpy as np
import faiss
from ..models.query import VectorQuery
from ..models.response import QueryResult

class VectorExecutor:
    def __init__(self, dimension: int, index_type: str = "L2"):
        self.dimension = dimension
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":  # Inner Product
            self.index = faiss.IndexFlatIP(dimension)
        self.id_mapping: Dict[int, Any] = {}

    def add_vectors(self, vectors: np.ndarray, ids: List[Any]):
        """Add vectors to the index with their corresponding IDs"""
        assert vectors.shape[1] == self.dimension
        vector_ids = np.arange(len(self.id_mapping), 
                             len(self.id_mapping) + len(vectors))
        self.index.add(vectors)
        
        # Update ID mapping
        for i, id_value in zip(vector_ids, ids):
            self.id_mapping[int(i)] = id_value

    async def execute(self, query: VectorQuery) -> QueryResult:
        try:
            vector = np.array([query.vector], dtype=np.float32)
            distances, indices = self.index.search(vector, query.k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid result
                    results.append({
                        "id": self.id_mapping.get(int(idx)),
                        "distance": float(distance),
                        "rank": i + 1
                    })

            return QueryResult(
                source="vector",
                data=results,
                metadata={"num_results": len(results)}
            )

        except Exception as e:
            return QueryResult(
                source="vector",
                data=[],
                metadata={"error": str(e)}
            ) 