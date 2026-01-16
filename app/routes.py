from flask import Blueprint, request, jsonify
from sqlalchemy.orm import Session
from .database import get_db
from .models.identity import QuizSession
from model import LDPredictor
import pandas as pd
import os

# Create a Blueprint for the API routes
api_bp = Blueprint('api', __name__)

# Load model
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

# Load model on import
load_model()

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

@api_bp.route('/session/create', methods=['POST'])
def create_session():
    """Create a new quiz session with hashed credential"""
    try:
        data = request.get_json()
        if not data or 'credential' not in data:
            return jsonify({'success': False, 'error': 'Credential required'}), 400
        
        credential = data['credential']
        
        db: Session = next(get_db())
        
        # Check if session already exists
        existing_session = db.query(QuizSession).filter(QuizSession.credential == credential).first()
        if existing_session:
            return jsonify({
                'success': True,
                'message': 'Session already exists',
                'has_data': existing_session.quiz_data is not None
            })
        
        # Create new session with empty data
        new_session = QuizSession(credential=credential, quiz_data=None)
        db.add(new_session)
        db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Session created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Process quiz data, make prediction, and save to database"""
    if predictor is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # Extract credential
        credential = data.get('credential')
        if not credential:
            return jsonify({'success': False, 'error': 'Credential required'}), 400
        
        # Extract quiz features (remove credential for prediction)
        quiz_features = {k: v for k, v in data.items() if k != 'credential'}
        
        # Create DataFrame for prediction
        df = pd.DataFrame([quiz_features])
        
        # Make prediction
        predictions = predictor.predict(df)
        result = predictions[0]
        
        # Prepare complete data to save (quiz features + prediction)
        complete_data = {
            'features': quiz_features,
            'prediction': result
        }
        
        # Save to database
        db: Session = next(get_db())
        session = db.query(QuizSession).filter(QuizSession.credential == credential).first()
        
        if not session:
            # Create session if it doesn't exist
            session = QuizSession(credential=credential, quiz_data=complete_data)
            db.add(session)
        else:
            # Update existing session
            session.quiz_data = complete_data
        
        db.commit()
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/session/<credential>', methods=['GET'])
def get_session(credential):
    """Retrieve quiz session data by credential"""
    try:
        db: Session = next(get_db())
        session = db.query(QuizSession).filter(QuizSession.credential == credential).first()
        
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'credential': session.credential,
            'quiz_data': session.quiz_data,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
