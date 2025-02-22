# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

import asyncio
import numpy as np
from unified_query.config import Settings
from unified_query.models.query import UnifiedQuery, SQLQuery, CypherQuery, VectorQuery
from unified_query.orchestrator import QueryOrchestrator

async def main():
    # Load settings from environment
    settings = Settings()
    
    # Initialize the orchestrator with settings
    orchestrator = QueryOrchestrator(
        sql_connection=settings.SQL_CONNECTION_STRING,
        neo4j_uri=settings.NEO4J_URI,
        neo4j_auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        vector_dimension=settings.VECTOR_DIMENSION
    )

    # Example 1: SQL query
    query = UnifiedQuery(
        sql=SQLQuery(
            query="SELECT id, name, age FROM users WHERE age > :age",
            parameters={"age": 25}
        ),
        cypher=CypherQuery(
            query="MATCH (u:User)-[:FRIEND]->(f) RETURN u.name as user, f.name as friend",
            parameters={}
        ),
        vector=VectorQuery(
            vector=[0.1] * 128,
            k=5
        )
    )

    # Execute the combined query
    response = await orchestrator.execute(query)
    
    print("Combined Query Results:")
    for result in response.results:
        print(f"\nSource: {result.source}")
        print("Data:", result.data)
        print("Metadata:", result.metadata)
    print(f"Total execution time: {response.execution_time:.2f} seconds")

    # Example 2: SQL-only query
    sql_query = UnifiedQuery(
        sql=SQLQuery(
            query="""
            SELECT department, COUNT(*) as employee_count 
            FROM employees 
            GROUP BY department
            ORDER BY employee_count DESC
            """,
            parameters={}
        )
    )

    sql_response = await orchestrator.execute(sql_query)
    print("\nSQL-only Query Results:")
    print(sql_response.results[0].data if sql_response.results else "No results")

    # Example 3: Vector similarity search
    # First, add some vectors to the index
    if orchestrator.vector_executor:
        # Create sample vectors
        sample_vectors = np.random.rand(10, 128).astype('float32')
        sample_ids = [f"doc_{i}" for i in range(10)]
        
        # Add vectors to the index
        orchestrator.vector_executor.add_vectors(sample_vectors, sample_ids)
        
        # Perform similarity search
        vector_query = UnifiedQuery(
            vector=VectorQuery(
                vector=sample_vectors[0].tolist(),  # Search for similar vectors to the first one
                k=3  # Get top 3 results
            )
        )
        
        vector_response = await orchestrator.execute(vector_query)
        print("\nVector Search Results:")
        print(vector_response.results[0].data if vector_response.results else "No results")

    # Example 4: Graph traversal with Cypher
    graph_query = UnifiedQuery(
        cypher=CypherQuery(
            query="""
            MATCH (u:User {name: $username})-[:FRIEND]->(friend)
            RETURN friend.name as friend_name, friend.age as friend_age
            """,
            parameters={"username": "John"}
        )
    )

    graph_response = await orchestrator.execute(graph_query)
    print("\nGraph Query Results:")
    print(graph_response.results[0].data if graph_response.results else "No results")

    # Clean up
    await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(main())