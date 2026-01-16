"""
Example API Client for LEADIS Predictor
Demonstrates the credential-based session workflow
"""
import requests
import hashlib
import json

BASE_URL = "http://localhost:5000"

def hash_credential(raw_credential: str) -> str:
    """Hash credential before sending to server"""
    return hashlib.sha256(raw_credential.encode()).hexdigest()

def create_session(credential: str):
    """Step 1: Create a new quiz session"""
    response = requests.post(
        f"{BASE_URL}/session/create",
        json={"credential": credential}
    )
    return response.json()

def submit_quiz_and_predict(credential: str, quiz_data: dict):
    """Step 2: Submit quiz data and get prediction"""
    # Add credential to the quiz data
    payload = {
        "credential": credential,
        **quiz_data
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    return response.json()

def get_session_data(credential: str):
    """Step 3: Retrieve stored session data"""
    response = requests.get(f"{BASE_URL}/session/{credential}")
    return response.json()

if __name__ == "__main__":
    print("=" * 60)
    print("LEADIS Predictor - API Client Example")
    print("=" * 60)
    
    # Example workflow
    raw_credential = "user123@example.com"  # This would come from your frontend auth
    hashed_credential = hash_credential(raw_credential)
    
    print(f"\n1. Creating session with credential: {hashed_credential[:16]}...")
    session_result = create_session(hashed_credential)
    print(json.dumps(session_result, indent=2))
    
    # Example quiz data (35 features)
    quiz_data = {
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
    
    print(f"\n2. Submitting quiz and getting prediction...")
    prediction_result = submit_quiz_and_predict(hashed_credential, quiz_data)
    print(json.dumps(prediction_result, indent=2))
    
    print(f"\n3. Retrieving stored session data...")
    stored_data = get_session_data(hashed_credential)
    print(json.dumps(stored_data, indent=2))
    
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
