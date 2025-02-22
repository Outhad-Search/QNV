# QNV (Query Nexus Vault)

A powerful middleware package that provides a unified interface for executing SQL, Cypher (Neo4j), and vector similarity queries. QNV acts as a central nexus for your data operations, allowing seamless integration between relational databases, graph databases, and vector similarity searches.

> ⚠️ **IMPORTANT**  
> We haven't uploaded complete code as this is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi


## 🌟 Features

- Unified query interface for multiple database types
- Concurrent execution of queries across different systems
- Async operations with built-in retry mechanisms
- Type-safe query building with Pydantic models
- Environment-based configuration
- Comprehensive logging and error handling

## 🎯 What Makes QNV Novel?

### 1. Unified Query Interface
Unlike traditional solutions that handle databases separately, QNV provides a single interface for:
- SQL queries (relational data)
- Cypher queries (graph relationships)
- Vector similarity searches (embeddings)

### 2. Concurrent Multi-Database Operations
```python
# Execute queries across different databases simultaneously
query = UnifiedQuery(
    sql=SQLQuery("SELECT * FROM users"),
    cypher=CypherQuery("MATCH (u:User) RETURN u"),
    vector=VectorQuery(vector=[0.1, 0.2, ...])
)
```

### 3. Smart Query Orchestration
- Automatic parallel execution
- Intelligent error handling
- Cross-database result merging
- Transaction management

### 4. Type-Safe Query Building
```python
from qnv.models import SQLQuery, CypherQuery, VectorQuery

# Type-safe query construction
sql_query = SQLQuery(
    query="SELECT * FROM users WHERE age > :age",
    parameters={"age": 25}
)
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install qnv
```

### 2. Configuration
Create a `.env` file:
```env
SQL_CONNECTION_STRING=postgresql+asyncpg://user:password@localhost:5432/dbname
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
VECTOR_DIMENSION=128
```

### 3. Basic Usage
```python
import asyncio
from qnv.config import Settings
from qnv.models import UnifiedQuery, SQLQuery
from qnv.orchestrator import QueryOrchestrator

async def main():
    # Initialize with settings
    settings = Settings()
    orchestrator = QueryOrchestrator(
        sql_connection=settings.SQL_CONNECTION_STRING,
        neo4j_uri=settings.NEO4J_URI,
        neo4j_auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        vector_dimension=settings.VECTOR_DIMENSION
    )

    # Create and execute a query
    query = UnifiedQuery(
        sql=SQLQuery(
            query="SELECT * FROM users WHERE age > :age",
            parameters={"age": 25}
        )
    )
    
    response = await orchestrator.execute(query)
    print(response.results)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔥 Advanced Usage

### 1. Combined Queries
```python
# Query across all systems
query = UnifiedQuery(
    sql=SQLQuery(
        query="SELECT id, name FROM users WHERE dept = :dept",
        parameters={"dept": "Engineering"}
    ),
    cypher=CypherQuery(
        query="MATCH (u:User)-[:MANAGES]->(t:Team) RETURN u.name, t.name",
        parameters={}
    ),
    vector=VectorQuery(
        vector=[0.1] * 128,
        k=5
    )
)
```

### 2. Vector Similarity Search
```python
# Add vectors to the index
vectors = np.random.rand(10, 128).astype('float32')
ids = [f"doc_{i}" for i in range(10)]
orchestrator.vector_executor.add_vectors(vectors, ids)

# Search similar vectors
vector_query = UnifiedQuery(
    vector=VectorQuery(
        vector=vectors[0].tolist(),
        k=3
    )
)
```

### 3. Graph Traversal
```python
# Complex graph query
graph_query = UnifiedQuery(
    cypher=CypherQuery(
        query="""
        MATCH (u:User {name: $username})-[:FRIEND*1..2]->(friend)
        RETURN friend.name, friend.age
        ORDER BY friend.age DESC
        """,
        parameters={"username": "John"}
    )
)
```

### 4. Error Handling
```python
from qnv.utils import execute_query_with_retry

result = await execute_query_with_retry(
    orchestrator,
    query,
    max_retries=3,
    retry_delay=1.0
)
```


## To test the project, you can use the following commands:
But before that you need to create the database and the tables and the relationships.
so for that use this @neo4j_test_command.md and @sql_test_command.md files.
```bash
python -m examples/basic_usage.py
```

## 🛠️ Use Cases

1. **Knowledge Graphs with Vector Search**
   - Store structured data in SQL
   - Maintain relationships in Neo4j
   - Enable semantic search with vectors

2. **Recommendation Systems**
   - User preferences in SQL
   - Social connections in Neo4j
   - Content similarity in vector space

3. **Data Analytics Platform**
   - Transactional data in SQL
   - Network analysis in Neo4j
   - Pattern matching with vectors

## 📊 Performance Considerations

- Queries are executed concurrently when possible
- Connection pooling for SQL and Neo4j
- Efficient vector indexing with FAISS
- Automatic retry mechanism for failed queries

## 🔒 Security

- Environment-based configuration
- Parameterized queries
- Connection encryption
- Credential management

## 🤝 Contributing

Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md).

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙋‍♂️ Support

- Create an issue for bug reports
- Join our [Discord community](https://discord.gg/qnv) soon
- Check our [Documentation](https://qnv.readthedocs.io) soon

## 🌟 Star Us!

If you find QNV helpful, please star us on GitHub! It helps us reach more developers.
