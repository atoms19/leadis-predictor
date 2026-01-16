from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from ..database import Base

class UserIdentity(Base):
    __tablename__ = "user_identity"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    parent_name = Column(String) # In a real app, this should be encrypted
    child_name = Column(String)  # In a real app, this should be encrypted
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UserIdentity(id={self.id}, email={self.email})>"
