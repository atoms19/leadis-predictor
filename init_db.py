"""
Database initialization script
Creates all tables defined in SQLAlchemy models
"""
from app.database import engine, Base
from app.models.identity import QuizSession

def init_database():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully!")
    print("\nTables created:")
    print("  - quiz_sessions (credential, quiz_data, created_at, updated_at)")

if __name__ == "__main__":
    init_database()
