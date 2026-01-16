"""
Example client to test the Flask API server
"""
import requests
import json

# Server URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint"""
    print("\n[1] Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_features():
    """Test features endpoint"""
    print("\n[2] Getting Required Features...")
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status: {response.status_code}")
    data = response.json()
    if data['success']:
        print(f"Total features: {data['total_features']}")
        print(f"Numerical: {len(data['numerical_features'])}")
        print(f"Categorical: {len(data['categorical_features'])}")

def test_model_info():
    """Test model info endpoint"""
    print("\n[3] Getting Model Information...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    data = response.json()
    if data['success']:
        print(f"PCA Categories: {len(data['pca_transformers'])}")
        print(f"Total PCA components: {sum(info['n_components'] for info in data['pca_transformers'].values())}")

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n[4] Testing Single Prediction...")
    
    # Sample input data (typical child profile)
    sample_data = {
        # Demographic
        "age_months": 72,
        "primary_language": 0,
        "schooling_type": 3,
        "gender": 0,
        
        # Developmental
        "multilingualExposure": 0,
        "multilingual_exposure": 0,
        "birthHistory": 0,
        "age_first_word_months": 12,
        "age_first_sentence_months": 24,
        "history_speech_therapy": 0,
        "history_motor_delay": 0,
        
        # Sensory
        "hearingStatus": 0,
        "hearing_concerns": 0,
        "visionStatus": 0,
        "vision_concerns": 0,
        
        # Family
        "family_learning_difficulty": 0,
        "family_adhd": 0,
        
        # Response metrics
        "mean_response_accuracy": 0.85,
        "response_accuracy_std": 0.12,
        "mean_response_time_ms": 2500,
        "response_time_std_ms": 800,
        
        # Task performance
        "task_completion_rate": 0.90,
        "task_abandonment_count": 2,
        "instruction_follow_accuracy": 0.88,
        
        # Attention
        "mean_focus_duration_sec": 180,
        "attention_span_average": 200,
        "random_interaction_rate": 0.05,
        
        # Memory
        "max_sequence_length": 6,
        
        # Visual
        "visual_search_time_ms": 3000,
        
        # Auditory
        "auditory_processing_accuracy": 0.85,
        "average_audio_replays": 1,
        "pref_auditory": 0.4,
        
        # Motor Coordination
        "hand_laterality_accuracy": 0.90,
        "finger_counting_accuracy": 0.95,
        "hand_position_accuracy": 0.88
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['success']:
        print(f"\nPrediction Results:")
        print(f"Sample ID: {data['prediction']['sample_id']}")
        print(f"\nRisk Scores:")
        for risk_name, risk_value in data['prediction']['targets'].items():
            risk_level = "LOW" if risk_value < 0.3 else "MODERATE" if risk_value < 0.6 else "HIGH"
            print(f"  {risk_name:30s}: {risk_value:.4f} ({risk_level})")
    else:
        print(f"Error: {data.get('error')}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n[5] Testing Batch Prediction...")
    
    # Multiple samples
    batch_data = {
        "samples": [
            {
                "age_months": 60, "primary_language": 0, "schooling_type": 2,
                "gender": 1, "multilingualExposure": 0, "multilingual_exposure": 0,
                "birthHistory": 0, "age_first_word_months": 12,
                "age_first_sentence_months": 24, "history_speech_therapy": 0,
                "history_motor_delay": 0, "hearingStatus": 0, "hearing_concerns": 0,
                "visionStatus": 0, "vision_concerns": 0, "family_learning_difficulty": 0,
                "family_adhd": 0, "mean_response_accuracy": 0.90,
                "response_accuracy_std": 0.10, "mean_response_time_ms": 2000,
                "response_time_std_ms": 500, "task_completion_rate": 0.95,
                "task_abandonment_count": 1, "instruction_follow_accuracy": 0.92,
                "mean_focus_duration_sec": 200, "attention_span_average": 220,
                "random_interaction_rate": 0.03, "max_sequence_length": 7,
                "visual_search_time_ms": 2500, "auditory_processing_accuracy": 0.90,
                "average_audio_replays": 0, "pref_auditory": 0.4,
                "hand_laterality_accuracy": 0.95, "finger_counting_accuracy": 0.98,
                "hand_position_accuracy": 0.93
            },
            {
                "age_months": 84, "primary_language": 0, "schooling_type": 3,
                "gender": 0, "multilingualExposure": 1, "multilingual_exposure": 1,
                "birthHistory": 1, "age_first_word_months": 18,
                "age_first_sentence_months": 30, "history_speech_therapy": 1,
                "history_motor_delay": 1, "hearingStatus": 0, "hearing_concerns": 0,
                "visionStatus": 1, "vision_concerns": 1, "family_learning_difficulty": 1,
                "family_adhd": 1, "mean_response_accuracy": 0.70,
                "response_accuracy_std": 0.20, "mean_response_time_ms": 4000,
                "response_time_std_ms": 1500, "task_completion_rate": 0.75,
                "task_abandonment_count": 8, "instruction_follow_accuracy": 0.70,
                "mean_focus_duration_sec": 100, "attention_span_average": 120,
                "random_interaction_rate": 0.25, "max_sequence_length": 4,
                "visual_search_time_ms": 8000, "auditory_processing_accuracy": 0.65,
                "average_audio_replays": 4, "pref_auditory": 0.6,
                "hand_laterality_accuracy": 0.60, "finger_counting_accuracy": 0.70,
                "hand_position_accuracy": 0.65
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['success']:
        print(f"\nProcessed {data['count']} samples")
        for i, pred in enumerate(data['predictions']):
            print(f"\nSample {i+1} - ID: {pred['sample_id']}")
            max_risk = max(pred['targets'].items(), key=lambda x: x[1])
            print(f"  Highest risk: {max_risk[0]} = {max_risk[1]:.4f}")
    else:
        print(f"Error: {data.get('error')}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Flask API Server")
    print("=" * 60)
    
    try:
        test_health()
        test_features()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to server.")
        print("Make sure the server is running: python server.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
