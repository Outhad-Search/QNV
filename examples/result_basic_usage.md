Combined Query Results:

Source: sql
Data: [{'id': 1, 'name': 'John Doe', 'age': 30}, {'id': 2, 'name': 'Jane Smith', 'age': 28}, {'id': 3, 'name': 'Bob Wilson', 'age': 35}]
Metadata: {'row_count': 3}

Source: cypher
Data: [{'user': 'John', 'friend': 'Jane'}]
Metadata: {'record_count': 1}

Source: vector
Data: []
Metadata: {'num_results': 0}
Total execution time: 0.07 seconds

SQL-only Query Results:
[{'department': 'Engineering', 'employee_count': 2}, {'department': 'Marketing', 'employee_count': 1}]

Vector Search Results:
[{'id': 'doc_0', 'distance': 0.0, 'rank': 1}, {'id': 'doc_7', 'distance': 19.589317321777344, 'rank': 2}, {'id': 'doc_3', 'distance': 20.298107147216797, 'rank': 3}]

Graph Query Results:
[{'friend_name': 'Jane', 'friend_age': 28}]