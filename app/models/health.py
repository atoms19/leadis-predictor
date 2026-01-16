from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import enum
from ..database import Base

class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer-not-to-say"

class Assessment(Base):
    __tablename__ = "assessment"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_hash = Column(String, index=True) # Anonymized identifier
    age_months = Column(Integer)
    gender = Column(Enum(Gender))
    raw_data = Column(JSON) # Stores the full input vector
    predictions = Column(JSON) # Stores the risk scores
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Assessment(id={self.id}, user_hash={self.user_hash})>"
