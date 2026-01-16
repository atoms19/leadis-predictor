import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class LD_EarlyDetectionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.target_column = 'diagnosis_label'
        
        # 1. Categorical Features
        self.categorical_features = [
            'child_gender', 'primary_language', 'secondary_language', 
            'environment_type', 'camera_used', 'microphone_used'
        ]
        
        # 2. Numerical Features
        self.numerical_features = [
            'child_age_months', 
            'total_questions_correct', 'avg_response_time_ms', 
            'attention_drop_events', 'gaze_shift_frequency',
            'speech_response_latency_ms', 'visual_prompt_success_rate',
            'language_risk_score', 'attention_risk_score'
        ]

    def _build_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight='balanced', 
                random_state=42, n_jobs=-1
            ))
        ])

    def train(self, data_folder='training_data'):
        """
        Reads ALL .csv files in the specified folder, combines them, and trains the model.
        """
        print(f"Looking for data in folder: '{data_folder}'...")
        
        # 1. Find all CSV files
        search_path = os.path.join(data_folder, "*.csv")
        all_files = glob.glob(search_path)
        
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in '{data_folder}'!")
            
        print(f"Found {len(all_files)} files: {[os.path.basename(f) for f in all_files]}")

        # 2. Read and Combine Data
        df_list = []
        for filename in all_files:
            try:
                temp_df = pd.read_csv(filename)
                # Quick check: Ensure essential columns exist
                if self.target_column in temp_df.columns:
                    df_list.append(temp_df)
                else:
                    print(f"WARNING: Skipping {filename} - missing target column '{self.target_column}'")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not df_list:
            raise ValueError("No valid training data could be loaded.")

        df = pd.concat(df_list, ignore_index=True)
        print(f"Combined dataset size: {df.shape[0]} rows.")

        # 3. Prepare Data
        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target_column]

        # Encode Labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # 4. Train
        print("Training model pipeline...")
        self.model = self._build_pipeline()
        self.model.fit(X_train, y_train)

        # 5. Evaluate
        print("\n--- Model Evaluation ---")
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        target_names = [str(cls) for cls in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=target_names))
    def predict_analysis(self, new_data_csv):
        """
        Returns a list of dictionaries, perfect for JSON/API responses.
        Each item contains the diagnosis and the percentage probability for ALL conditions.
        """
        if self.model is None:
            raise Exception("Model not trained! Call load_model() first.")

        # 1. Load and Preprocess
        df_new = pd.read_csv(new_data_csv)
        
        # Ensure we only select columns the model knows about
        # (This handles cases where the CSV has extra 'metadata' columns)
        X_new = df_new[self.numerical_features + self.categorical_features]
        
        # 2. Get Raw Probabilities (The "Percentages")
        # Returns an array like [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]]
        probs = self.model.predict_proba(X_new)
        
        # 3. Get Hard Predictions (The "Labels")
        predictions_encoded = self.model.predict(X_new)
        prediction_labels = self.label_encoder.inverse_transform(predictions_encoded)
        
        # 4. Format the Output
        api_response = []
        
        # Get the list of all possible disabilities the model knows
        all_conditions = self.label_encoder.classes_
        
        for i, row_id in enumerate(df_new['child_id']):
            # Create the probability dictionary: {'ADHD': 12.5, 'Dyslexia': 80.1}
            risk_profile = {}
            for j, condition in enumerate(all_conditions):
                risk_profile[condition] = round(probs[i][j] * 100, 2) # Convert 0.125 -> 12.5%

            # Build the final object for this child
            child_result = {
                "child_id": str(row_id),
                "predicted_diagnosis": prediction_labels[i],
                "confidence_score": risk_profile[prediction_labels[i]], # Probability of the winner
                "risk_breakdown": risk_profile # <--- This is the dictionary you asked for
            }
            api_response.append(child_result)
            
        return api_response
    def save_model(self, filename='ld_model.pkl'):
        joblib.dump({'pipeline': self.model, 'encoder': self.label_encoder}, filename)
        print(f"Model saved to {filename}")

# --- HELPER: Generate Dummy Files for Testing ---
def generate_dummy_data_folder(folder_name='training_data'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Generate 3 different "clinic" files to simulate batch data
    filenames = ['clinic_north.csv', 'clinic_south.csv', 'home_sessions.csv']
    
    for fname in filenames:
        path = os.path.join(folder_name, fname)
        # Create 200 rows per file
        # (Reusing the generator logic from before, shortened for brevity)
        rows = 200
        data = {
            'child_id': [f'C_{fname}_{i}' for i in range(rows)],
            'child_gender': np.random.choice(['M', 'F'], rows),
            'primary_language': np.random.choice(['English', 'Spanish'], rows),
            'secondary_language': np.random.choice(['None', 'French'], rows),
            'environment_type': np.random.choice(['Home', 'Clinic'], rows),
            'camera_used': np.random.choice([0, 1], rows),
            'microphone_used': np.random.choice([0, 1], rows),
            'child_age_months': np.random.randint(48, 96, rows),
            'total_questions_correct': np.random.randint(0, 50, rows),
            'avg_response_time_ms': np.random.randint(500, 5000, rows),
            'attention_drop_events': np.random.randint(0, 20, rows),
            'gaze_shift_frequency': np.random.uniform(0.1, 0.9, rows),
            'speech_response_latency_ms': np.random.randint(200, 2000, rows),
            'visual_prompt_success_rate': np.random.uniform(0.0, 1.0, rows),
            'language_risk_score': np.random.uniform(0, 1, rows),
            'attention_risk_score': np.random.uniform(0, 1, rows),
            'diagnosis_label': np.random.choice(['No Risk', 'ADHD', 'Dyslexia'], rows)
        }
        # Inject signal
        df = pd.DataFrame(data)
        mask_adhd = df['attention_drop_events'] > 15
        df.loc[mask_adhd, 'diagnosis_label'] = 'ADHD'
        df.to_csv(path, index=False)
        print(f"Created: {path}")

# --- MAIN ---
if __name__ == "__main__":
    # 1. Setup Folders & Data
    generate_dummy_data_folder('training_data')
    
    # 2. Initialize
    system = LD_EarlyDetectionModel()
    
    # 3. Train (Automatically scans the folder)
    try:
        system.train('training_data') 
        system.save_model()
        print("\nSUCCESS: Model trained on all CSVs in folder.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
