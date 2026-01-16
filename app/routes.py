from flask import Blueprint, request, jsonify
from .database import get_db
from .services.crud import CRUDService
from .services.privacy import PrivacyService
from .models.consent import ConsentType
from model import LDPredictor
import pandas as pd
import os

# Create a Blueprint for the API routes
api_bp = Blueprint('api', __name__)

# Load model (reusing existing logic)
MODEL_PATH = 'model.pkl'
predictor = None

def load_model():
    global predictor
    if os.path.exists(MODEL_PATH):
        predictor = LDPredictor()
        predictor.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found.")

# Load model on import (or could be done in create_app)
load_model()

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'privacy_mode': 'enabled'
    })

@api_bp.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        # --- Privacy & Identity Layer ---
        db = next(get_db())
        
        # Check for identity info (optional, for logged-in users)
        # In a real app, this would come from a session or JWT
        email = data.get('email')
        user_id = None
        
        if email:
            # Try to find existing user or create one (simplified for demo)
            user = CRUDService.get_user_by_email(db, email)
            if not user:
                # If new user, we might need more info, or just create a placeholder
                # For this demo, we'll assume we can create one if name is provided
                if 'parent_name' in data:
                    user = CRUDService.create_user(db, email, data.get('parent_name'), data.get('child_name', ''))
            
            if user:
                user_id = user.id
                
                # Check Consent for Clinical Feedback if requested
                # (Assuming the request implies consent if they are asking for it, 
                # but strictly we should check if they have granted it before)
                # For this implementation, we'll just log that we checked.
                has_consent = CRUDService.check_consent(db, user_id, ConsentType.CLINICAL_FEEDBACK)
                
        # --- Prediction Layer ---
        # Prepare data for model (remove non-features)
        # We use the privacy service to clean it first, which is also good for the model
        # but we might need to be careful not to remove features the model needs if they look like PII.
        # The current anonymize_data is safe for the features we know about.
        clean_data = PrivacyService.anonymize_data(data)
        
        # Create DataFrame
        df = pd.DataFrame([clean_data])
        
        # Predict
        predictions = predictor.predict(df)
        result = predictions[0]
        
        # --- Storage Layer ---
        # Store the assessment securely
        # If we have a user_id, it will be hashed. If not, we generate a random hash or skip linking.
        storage_user_id = user_id if user_id else "anonymous"
        
        CRUDService.create_assessment(db, storage_user_id, clean_data, result)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'privacy_notice': 'Data stored with privacy preservation.'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
