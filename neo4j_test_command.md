> ⚠️ **IMPORTANT**  
> This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

// First, clear any existing data
MATCH (n) DETACH DELETE n;

// Create John node
CREATE (john:User {name: 'John', age: 30});

// Create Jane node
CREATE (jane:User {name: 'Jane', age: 28});

// Create relationship between John and Jane
MATCH (john:User {name: 'John'}), (jane:User {name: 'Jane'})
CREATE (john)-[:FRIEND]->(jane);

// Verify the data
MATCH (u:User)-[r:FRIEND]->(f:User)
RETURN u.name as user, f.name as friend;