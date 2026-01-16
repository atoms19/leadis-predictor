import hashlib
import json

class PrivacyService:
    @staticmethod
    def hash_user_id(user_id: str, salt: str = "some_secret_salt") -> str:
        """
        Generates a deterministic hash for linking health data without exposing identity.
        In a real system, the salt should be a securely stored environment variable.
        """
        return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()

    @staticmethod
    def anonymize_data(data: dict) -> dict:
        """
        Strips PII from datasets for ML training.
        Removes keys that are not in the allowed list of features.
        """
        # This is a basic implementation. 
        # In a real scenario, we would have a strict allowlist of fields.
        # For now, we assume the input 'data' is already mostly features, 
        # but we ensure no obvious PII fields are present if they were passed.
        
        pii_fields = ['name', 'email', 'phone', 'address', 'dob', 'parent_name', 'child_name']
        anonymized = data.copy()
        for field in pii_fields:
            if field in anonymized:
                del anonymized[field]
        
        return anonymized
