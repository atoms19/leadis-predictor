import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models.identity import UserIdentity
from app.models.consent import UserConsent, ConsentType
from app.models.health import Assessment, Gender
from app.services.crud import CRUDService
from app.services.privacy import PrivacyService
import os

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_session():
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_create_user(db_session):
    user = CRUDService.create_user(db_session, "test@example.com", "Parent", "Child")
    assert user.email == "test@example.com"
    assert user.id is not None

def test_consent_flow(db_session):
    user = CRUDService.create_user(db_session, "consent@example.com", "P", "C")
    
    # Initially no consent
    assert CRUDService.check_consent(db_session, user.id, ConsentType.CLINICAL_FEEDBACK) is False
    
    # Grant consent
    CRUDService.create_consent(db_session, user.id, ConsentType.CLINICAL_FEEDBACK, True, "127.0.0.1")
    assert CRUDService.check_consent(db_session, user.id, ConsentType.CLINICAL_FEEDBACK) is True
    
    # Revoke consent
    CRUDService.create_consent(db_session, user.id, ConsentType.CLINICAL_FEEDBACK, False, "127.0.0.1")
    assert CRUDService.check_consent(db_session, user.id, ConsentType.CLINICAL_FEEDBACK) is False

def test_assessment_privacy(db_session):
    user = CRUDService.create_user(db_session, "privacy@example.com", "P", "C")
    
    data = {
        "age_months": 48,
        "gender": 0, # Male
        "name": "Should Be Removed",
        "email": "remove@me.com",
        "some_feature": 123
    }
    predictions = {"risk": 0.5}
    
    assessment = CRUDService.create_assessment(db_session, user.id, data, predictions)
    
    # Verify linking via hash
    expected_hash = PrivacyService.hash_user_id(user.id)
    assert assessment.user_hash == expected_hash
    
    # Verify PII removal
    assert "name" not in assessment.raw_data
    assert "email" not in assessment.raw_data
    assert assessment.raw_data["some_feature"] == 123
    
    # Verify gender mapping
    assert assessment.gender == Gender.MALE

def test_anonymization_logic():
    data = {"name": "Test", "age": 10, "email": "test@test.com"}
    clean = PrivacyService.anonymize_data(data)
    assert "name" not in clean
    assert "email" not in clean
    assert "age" in clean
