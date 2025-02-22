> ⚠️ **IMPORTANT**  
> This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

CREATE DATABASE unified_query_db;
\c unified_query_db

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(100)
);

INSERT INTO users (name, age) VALUES 
    ('John Doe', 30),
    ('Jane Smith', 28),
    ('Bob Wilson', 35);

INSERT INTO employees (name, department) VALUES 
    ('John Doe', 'Engineering'),
    ('Jane Smith', 'Marketing'),
    ('Bob Wilson', 'Engineering');