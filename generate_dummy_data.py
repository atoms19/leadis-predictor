import pandas as pd
import numpy as np
import os

def generate_dummy_data(num_samples=200, output_file='training_data/dummy_new.csv'):
    np.random.seed(42)
    
    data = {
        # Metadata
        'sample_id': [f'S_{i}' for i in range(num_samples)],
        'session_id': [f'SES_{i}' for i in range(num_samples)],
        'timestamp_utc': pd.date_range(start='2025-01-01', periods=num_samples, freq='h').strftime('%Y-%m-%dT%H:%M:%S'),
        'data_version': ['v1.0'] * num_samples,

        # Child Metadata
        'age_months': np.random.randint(48, 120, num_samples),
        'primary_language': np.random.choice(['English', 'Spanish', 'Hindi'], num_samples),
        'multilingual_exposure': np.random.choice([0, 1], num_samples),
        'schooling_type': np.random.choice(['Public', 'Private', 'Homeschool'], num_samples),
        'gender': np.random.choice(['M', 'F'], num_samples),

        # Parent History
        'age_first_word_months': np.random.randint(10, 24, num_samples),
        'age_first_sentence_months': np.random.randint(18, 36, num_samples),
        'history_speech_therapy': np.random.choice([0, 1], num_samples),
        'history_motor_delay': np.random.choice([0, 1], num_samples),
        'vision_concerns': np.random.choice([0, 1], num_samples),
        'hearing_concerns': np.random.choice([0, 1], num_samples),
        'family_learning_difficulty': np.random.choice([0, 1], num_samples),
        'family_adhd': np.random.choice([0, 1], num_samples),

        # Interaction Metrics
        'mean_response_accuracy': np.random.uniform(0.5, 1.0, num_samples),
        'response_accuracy_std': np.random.uniform(0.0, 0.2, num_samples),
        'mean_response_time_ms': np.random.uniform(500, 5000, num_samples),
        'response_time_std_ms': np.random.uniform(100, 1000, num_samples),
        'instruction_repeat_count': np.random.randint(0, 5, num_samples),
        'guessing_rate': np.random.uniform(0.0, 0.5, num_samples),

        # Attention Features
        'task_completion_rate': np.random.uniform(0.7, 1.0, num_samples),
        'mean_focus_duration_sec': np.random.uniform(10, 60, num_samples),
        'attention_dropoff_slope': np.random.uniform(-0.1, 0.0, num_samples),
        'random_interaction_rate': np.random.uniform(0.0, 0.3, num_samples),
        'task_abandonment_count': np.random.randint(0, 3, num_samples),

        # Memory Features
        'immediate_recall_accuracy': np.random.uniform(0.4, 1.0, num_samples),
        'delayed_recall_accuracy': np.random.uniform(0.3, 1.0, num_samples),
        'max_sequence_length': np.random.randint(3, 9, num_samples),
        'sequence_order_error_rate': np.random.uniform(0.0, 0.4, num_samples),
        'memory_omission_rate': np.random.uniform(0.0, 0.3, num_samples),

        # Visual Features
        'shape_discrimination_accuracy': np.random.uniform(0.6, 1.0, num_samples),
        'mirror_confusion_rate': np.random.uniform(0.0, 0.4, num_samples),
        'visual_search_time_ms': np.random.uniform(1000, 8000, num_samples),
        'pattern_completion_accuracy': np.random.uniform(0.5, 1.0, num_samples),
        'visual_error_consistency': np.random.uniform(0.0, 0.5, num_samples),

        # Motor Features
        'instruction_follow_accuracy': np.random.uniform(0.6, 1.0, num_samples),
        'movement_initiation_latency_ms': np.random.uniform(200, 1000, num_samples),
        'movement_smoothness_variance': np.random.uniform(0.0, 0.5, num_samples),
        'left_right_confusion_rate': np.random.uniform(0.0, 0.3, num_samples),
        'bilateral_coordination_score': np.random.uniform(0.4, 1.0, num_samples),

        # Speech Features
        'vocabulary_diversity': np.random.uniform(0.5, 1.0, num_samples),
        'mean_utterance_length': np.random.uniform(2.0, 10.0, num_samples),
        'speech_rate_wpm': np.random.uniform(80, 150, num_samples),
        'pronunciation_error_rate': np.random.uniform(0.0, 0.3, num_samples),
        'hesitation_frequency': np.random.uniform(0.0, 0.4, num_samples),

        # Reading Features
        'reading_speed_wpm': np.random.uniform(50, 200, num_samples),
        'word_skip_rate': np.random.uniform(0.0, 0.2, num_samples),
        'line_regression_rate': np.random.uniform(0.0, 0.3, num_samples),
        'letter_reversal_rate': np.random.uniform(0.0, 0.2, num_samples),
        'audio_text_mismatch_rate': np.random.uniform(0.0, 0.3, num_samples),

        # Learning Preference
        'pref_visual': np.random.uniform(0.0, 1.0, num_samples),
        'pref_auditory': np.random.uniform(0.0, 1.0, num_samples),
        'pref_kinesthetic': np.random.uniform(0.0, 1.0, num_samples),

        # Engineered Features
        'visual_vs_audio_accuracy_delta': np.random.uniform(-0.5, 0.5, num_samples),
        'audio_vs_text_latency_delta': np.random.uniform(-500, 500, num_samples),
        'motor_vs_visual_performance_gap': np.random.uniform(0.0, 0.5, num_samples),
        'attention_vs_accuracy_correlation': np.random.uniform(-1.0, 1.0, num_samples),
        'intra_session_consistency_score': np.random.uniform(0.5, 1.0, num_samples),

        # Targets (Risk Scores)
        'risk_reading': np.random.uniform(0.0, 1.0, num_samples),
        'risk_writing': np.random.uniform(0.0, 1.0, num_samples),
        'risk_attention': np.random.uniform(0.0, 1.0, num_samples),
        'risk_working_memory': np.random.uniform(0.0, 1.0, num_samples),
        'risk_expressive_language': np.random.uniform(0.0, 1.0, num_samples),
        'risk_receptive_language': np.random.uniform(0.0, 1.0, num_samples),
        'risk_visual_processing': np.random.uniform(0.0, 1.0, num_samples),
        'risk_motor_coordination': np.random.uniform(0.0, 1.0, num_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Inject some correlations for realism
    # High attention risk -> low task completion
    mask_attn = df['risk_attention'] > 0.7
    df.loc[mask_attn, 'task_completion_rate'] = df.loc[mask_attn, 'task_completion_rate'] * 0.5
    
    # High reading risk -> low reading speed
    mask_read = df['risk_reading'] > 0.7
    df.loc[mask_read, 'reading_speed_wpm'] = df.loc[mask_read, 'reading_speed_wpm'] * 0.5
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    df.to_csv(output_file, index=False)
    print(f"Generated {output_file} with {num_samples} samples.")

if __name__ == "__main__":
    generate_dummy_data()
