from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from ..database import Base

class QuizSession(Base):
    """Simplified session model with credential-based storage"""
    __tablename__ = "quiz_sessions"

    credential = Column(String, primary_key=True)  # Hashed credential from frontend
    quiz_data = Column(JSON, nullable=True)  # JSON with quiz answers and prediction result
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<QuizSession(credential={self.credential[:8]}..., has_data={self.quiz_data is not None})>"
