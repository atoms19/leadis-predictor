# Database Setup Guide

## PostgreSQL Database Schema

The LEADIS Predictor uses a simplified PostgreSQL schema designed for credential-based quiz sessions.

### Schema Design

**Table: `quiz_sessions`**

| Column | Type | Description |
|--------|------|-------------|
| `credential` | VARCHAR (PRIMARY KEY) | Hashed credential from frontend (e.g., SHA-256 hash) |
| `quiz_data` | JSON | Complete quiz responses and prediction results |
| `created_at` | TIMESTAMP | Session creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

### Data Flow Pattern

1. **Frontend sends hashed credential** → Server creates empty session row
2. **User completes quiz** → Frontend sends credential + quiz data
3. **Server processes prediction** → Saves quiz data + results as JSON
4. **Data persists** → Available for retrieval using credential

## Setup Instructions

### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download from [postgresql.org](https://www.postgresql.org/download/windows/)

### 2. Create Database

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Run the setup script
\i setup_database.sql

# Or manually:
CREATE DATABASE leadis_db;
CREATE USER leadis_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE leadis_db TO leadis_user;
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
DB_USER=leadis_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=leadis_db
```

Or set environment variables directly:

```bash
export DB_USER=leadis_user
export DB_PASSWORD=your_secure_password
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=leadis_db
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Initialize Database Tables

```bash
python init_db.py
```

This creates the `quiz_sessions` table using SQLAlchemy.

## API Endpoints

### 1. Create Session
**POST** `/session/create`

Creates a new quiz session with a hashed credential.

**Request:**
```json
{
  "credential": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Session created successfully"
}
```

### 2. Submit Quiz & Get Prediction
**POST** `/predict`

Submits quiz data, gets prediction, and saves everything to database.

**Request:**
```json
{
  "credential": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
  "age_months": 72,
  "primary_language": 0,
  "schooling_type": 3,
  ... (all 35 features)
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "sample_id": "sample_0",
    "targets": {
      "risk_reading": 0.1234,
      "risk_writing": 0.2345,
      "risk_attention": 0.1567,
      "risk_working_memory": 0.2890,
      "risk_receptive_language": 0.1456,
      "risk_visual_processing": 0.2123,
      "risk_motor_coordination": 0.1678
    }
  }
}
```

**Stored in Database:**
```json
{
  "features": { ... all 35 quiz features ... },
  "prediction": {
    "sample_id": "sample_0",
    "targets": { ... 7 risk scores ... }
  }
}
```

### 3. Retrieve Session Data
**GET** `/session/<credential>`

Retrieves stored quiz data and prediction results.

**Response:**
```json
{
  "success": true,
  "credential": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
  "quiz_data": {
    "features": { ... },
    "prediction": { ... }
  },
  "created_at": "2026-01-17T10:30:00",
  "updated_at": "2026-01-17T10:35:00"
}
```

## Example Client Usage

See `example_client.py` for a complete workflow example:

```bash
python example_client.py
```

## Security Notes

1. **Credential Hashing**: Always hash credentials on the frontend before sending to the server
2. **HTTPS**: Use HTTPS in production to encrypt data in transit
3. **Database Credentials**: Store database credentials in environment variables, never in code
4. **Access Control**: Implement rate limiting and authentication as needed
5. **Data Retention**: Consider implementing a data retention policy for GDPR compliance

## Troubleshooting

### Connection Error
```
sqlalchemy.exc.OperationalError: could not connect to server
```
**Solution:** Check that PostgreSQL is running and credentials are correct.

### Authentication Failed
```
FATAL: password authentication failed for user "leadis_user"
```
**Solution:** Verify DB_PASSWORD environment variable matches the database user password.

### Table Already Exists
```
Table 'quiz_sessions' already exists
```
**Solution:** This is normal. SQLAlchemy creates tables if they don't exist.

## Migration from SQLite

If you were previously using SQLite, you can migrate data:

```bash
# Export from SQLite
sqlite3 leadis.db .dump > data.sql

# Import to PostgreSQL (requires conversion)
# Use tools like pgloader or manually convert
```

## Production Deployment

For production:

1. Use managed PostgreSQL (AWS RDS, Google Cloud SQL, Azure Database)
2. Enable SSL connections
3. Set up database backups
4. Configure connection pooling
5. Use read replicas for scaling

## Schema Advantages

✓ **Simple**: One table with minimal columns  
✓ **Flexible**: JSON column stores any quiz structure  
✓ **Fast**: Primary key lookup by credential  
✓ **Scalable**: PostgreSQL handles millions of rows  
✓ **Privacy**: Credentials are already hashed by frontend
