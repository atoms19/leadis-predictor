from sqlalchemy.orm import Session
from ..models.identity import UserIdentity
from ..models.consent import UserConsent, ConsentType
from ..models.health import Assessment, Gender
from .privacy import PrivacyService
import uuid

class CRUDService:
    @staticmethod
    def create_user(db: Session, email: str, parent_name: str, child_name: str) -> UserIdentity:
        db_user = UserIdentity(email=email, parent_name=parent_name, child_name=child_name)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> UserIdentity:
        return db.query(UserIdentity).filter(UserIdentity.email == email).first()

    @staticmethod
    def create_consent(db: Session, user_id: str, consent_type: ConsentType, is_granted: bool, ip_address: str) -> UserConsent:
        db_consent = UserConsent(
            user_id=user_id,
            consent_type=consent_type,
            is_granted=is_granted,
            ip_address=ip_address
        )
        db.add(db_consent)
        db.commit()
        db.refresh(db_consent)
        return db_consent

    @staticmethod
    def check_consent(db: Session, user_id: str, consent_type: ConsentType) -> bool:
        consent = db.query(UserConsent).filter(
            UserConsent.user_id == user_id,
            UserConsent.consent_type == consent_type
        ).order_by(UserConsent.timestamp.desc()).first()
        
        if consent:
            return consent.is_granted
        return False

    @staticmethod
    def create_assessment(db: Session, user_id: str, data: dict, predictions: dict) -> Assessment:
        # 1. Hash the user ID
        user_hash = PrivacyService.hash_user_id(user_id)
        
        # 2. Extract relevant fields for the Assessment table
        # Note: In a real app, we might validate these against the schema
        age_months = data.get('age_months')
        gender_str = data.get('gender')
        
        # Map gender string to Enum if possible, else None or default
        gender_enum = None
        if gender_str == 0: # Based on field-mapping.json: male=0
            gender_enum = Gender.MALE
        elif gender_str == 1: # female=1
            gender_enum = Gender.FEMALE
        # ... handle other cases as needed
        
        # 3. Anonymize the raw data before storage (double check)
        anonymized_data = PrivacyService.anonymize_data(data)
        
        db_assessment = Assessment(
            user_hash=user_hash,
            age_months=age_months,
            gender=gender_enum,
            raw_data=anonymized_data,
            predictions=predictions
        )
        db.add(db_assessment)
        db.commit()
        db.refresh(db_assessment)
        return db_assessment
