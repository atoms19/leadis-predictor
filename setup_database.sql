-- PostgreSQL Database Setup for LEADIS Predictor
-- Run this script to create the database and user

-- Create database
CREATE DATABASE leadis_db;

-- Create user (optional, change password)
CREATE USER leadis_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE leadis_db TO leadis_user;

-- Connect to the database
\c leadis_db

-- The quiz_sessions table will be created automatically by SQLAlchemy
-- But here's the schema for reference:

CREATE TABLE IF NOT EXISTS quiz_sessions (
    credential VARCHAR PRIMARY KEY,
    quiz_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_created_at ON quiz_sessions(created_at);
