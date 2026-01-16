from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from ..database import Base

class ConsentType(enum.Enum):
    CLINICAL_FEEDBACK = "clinical_feedback"
    MODEL_TRAINING = "model_training"

class UserConsent(Base):
    __tablename__ = "user_consent"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_identity.id"))
    consent_type = Column(Enum(ConsentType))
    is_granted = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)

    user = relationship("UserIdentity")

    def __repr__(self):
        return f"<UserConsent(user_id={self.user_id}, type={self.consent_type}, granted={self.is_granted})>"
