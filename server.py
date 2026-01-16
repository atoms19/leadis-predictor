"""
Flask API Server for Learning Disability Risk Prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from model import LDPredictor
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model at startup
MODEL_PATH = 'model.pkl'
predictor = None

def load_model():
    """Load the trained model"""
    global predictor
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first using: python train.py")
    
    predictor = LDPredictor()
    predictor.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")

# Load model when server starts
try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("The server will start but predictions will fail until a model is loaded.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expects JSON body with all required features:
    {
        "age_months": 72,
        "primary_language": 0,
        "schooling_type": 3,
        "gender": 0,
        "multilingualExposure": 0,
        "multilingual_exposure": 0,
        "birthHistory": 0,
        "age_first_word_months": 12,
        "age_first_sentence_months": 24,
        "history_speech_therapy": 0,
        "history_motor_delay": 0,
        "hearingStatus": 0,
        "hearing_concerns": 0,
        "visionStatus": 0,
        "vision_concerns": 0,
        "family_learning_difficulty": 0,
        "family_adhd": 0,
        "mean_response_accuracy": 0.85,
        "response_accuracy_std": 0.12,
        "mean_response_time_ms": 2500,
        "response_time_std_ms": 800,
        "task_completion_rate": 0.90,
        "task_abandonment_count": 2,
        "instruction_follow_accuracy": 0.88,
        "mean_focus_duration_sec": 180,
        "attention_span_average": 200,
        "random_interaction_rate": 0.05,
        "max_sequence_length": 6,
        "visual_search_time_ms": 3000,
        "auditory_processing_accuracy": 0.85,
        "average_audio_replays": 1,
        "pref_auditory": 0.4,
        "hand_laterality_accuracy": 0.90,
        "finger_counting_accuracy": 0.95,
        "hand_position_accuracy": 0.88
    }
    
    Returns:
    {
        "success": true,
        "predictions": {
            "sample_id": "sample_0",
            "targets": {
                "risk_reading": 0.1234,
                "risk_writing": 0.2345,
                ...
            }
        }
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required features
        required_numerical = predictor.numerical_features
        required_categorical = predictor.categorical_features
        all_required = required_numerical + required_categorical
        
        missing_features = [f for f in all_required if f not in data]
        
        if missing_features:
            return jsonify({
                'success': False,
                'error': 'Missing required features',
                'missing_features': missing_features
            }), 400
        
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Make prediction
        predictions = predictor.predict(df)
        
        return jsonify({
            'success': True,
            'prediction': predictions[0]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expects JSON body with array of feature objects:
    {
        "samples": [
            { ...features... },
            { ...features... }
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": [
            {
                "sample_id": "sample_0",
                "targets": { ... }
            },
            ...
        ]
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({
                'success': False,
                'error': 'No samples provided. Expected format: {"samples": [...]}'
            }), 400
        
        samples = data['samples']
        
        if not isinstance(samples, list):
            return jsonify({
                'success': False,
                'error': 'Samples must be an array'
            }), 400
        
        # Create DataFrame from input data
        df = pd.DataFrame(samples)
        
        # Make predictions
        predictions = predictor.predict(df)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """
    Get list of required features
    
    Returns:
    {
        "numerical_features": [...],
        "categorical_features": [...],
        "target_columns": [...]
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'numerical_features': predictor.numerical_features,
        'categorical_features': predictor.categorical_features,
        'target_columns': predictor.target_columns,
        'total_features': len(predictor.numerical_features) + len(predictor.categorical_features)
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Get model information
    
    Returns model metadata and PCA configuration
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Get PCA information
    pca_info = {}
    for cat_name, pca in predictor.pca_transformers.items():
        pca_info[cat_name] = {
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(pca.explained_variance_ratio_.sum())
        }
    
    return jsonify({
        'success': True,
        'model_path': MODEL_PATH,
        'n_components_per_category': predictor.n_components_per_category,
        'pca_transformers': pca_info,
        'numerical_features_count': len(predictor.numerical_features),
        'categorical_features_count': len(predictor.categorical_features),
        'target_columns': predictor.target_columns
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Learning Disability Risk Prediction API Server")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /features         - Get required features")
    print("  GET  /model/info       - Get model information")
    print("  POST /predict          - Single prediction")
    print("  POST /predict/batch    - Batch predictions")
    print("\n" + "=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
