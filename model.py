import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LDPredictor:
    def __init__(self):
        self.model = None
        
        # Targets
        self.target_columns = [
            'risk_reading', 'risk_writing', 'risk_attention', 
            'risk_working_memory', 'risk_expressive_language', 
            'risk_receptive_language', 'risk_visual_processing', 
            'risk_motor_coordination'
        ]
        
        # 1. Categorical Features
        self.categorical_features = [
            'primary_language', 'schooling_type', 'gender',
            'multilingualExposure', 'birthHistory', 'hearingStatus', 'visionStatus'
        ]
        
        # 2. Numerical Features
        self.numerical_features = [
            'age_months', 'multilingual_exposure', 
            'age_first_word_months', 'age_first_sentence_months',
            'history_speech_therapy', 'history_motor_delay', 
            'vision_concerns', 'hearing_concerns', 
            'family_learning_difficulty', 'family_adhd',
            'mean_response_accuracy', 'response_accuracy_std', 
            'mean_response_time_ms', 'response_time_std_ms',
            'task_completion_rate', 'mean_focus_duration_sec', 
            'attention_dropoff_slope', 'random_interaction_rate', 
            'task_abandonment_count', 'max_sequence_length',
            'sequence_order_error_rate', 'visual_search_time_ms',
            'instruction_follow_accuracy', 'left_right_confusion_rate', 
            'speech_rate_wpm', 'auditory_processing_accuracy',
            'average_audio_replays', 'hesitation_frequency',
            'reading_speed_wpm', 'reading_accuracy',
            'letter_reversal_rate', 'audio_text_mismatch_rate',
            'pref_visual', 'pref_auditory', 'attention_span_average'
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

        # Using MultiOutputRegressor with GradientBoostingRegressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))
        ])
        
        return pipeline

    def train(self, df):
        """
        Trains the model using GridSearchCV for hyperparameter tuning.
        """
        print("Preprocessing data...")
        
        # Filter columns that exist in the dataframe
        available_num = [c for c in self.numerical_features if c in df.columns]
        available_cat = [c for c in self.categorical_features if c in df.columns]
        
        # Update features based on what's available (robustness)
        self.numerical_features = available_num
        self.categorical_features = available_cat
        
        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target_columns]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build Pipeline
        pipeline = self._build_pipeline()

        # Hyperparameter Tuning (Simplified for MultiOutput)
        # Note: GridSearchCV with MultiOutputRegressor can be heavy. 
        # We'll tune the base estimator.
        param_grid = {
            'regressor__estimator__n_estimators': [100],
            'regressor__estimator__learning_rate': [0.1],
            'regressor__estimator__max_depth': [3]
        }

        print("Starting Grid Search...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

        # Evaluate
        print("\n--- Model Evaluation ---")
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Per-target evaluation
        print("\nPer-Target R2 Scores:")
        for i, target in enumerate(self.target_columns):
            score = r2_score(y_test.iloc[:, i], y_pred[:, i])
            print(f"  {target}: {score:.4f}")

    def predict(self, df_new):
        """
        Returns a list of dictionaries with predicted risk scores.
        """
        if self.model is None:
            raise Exception("Model not trained! Call load() first.")

        # Ensure we only select columns the model knows about
        X_new = df_new[self.numerical_features + self.categorical_features]
        
        # Predict
        predictions = self.model.predict(X_new)
        
        # Format the Output
        results = []
        
        for i in range(len(df_new)):
            row_id = df_new.iloc[i].get('sample_id', f'Unknown_{i}')
            
            # Map predictions to target names
            targets_pred = {}
            for j, target in enumerate(self.target_columns):
                # Clip values to 0-1 range just in case
                val = float(predictions[i][j])
                val = max(0.0, min(1.0, val))
                targets_pred[target] = round(val, 4)

            child_result = {
                "sample_id": str(row_id),
                "targets": targets_pred
            }
            results.append(child_result)
            
        return results

    def save(self, filename='model.pkl'):
        joblib.dump({
            'pipeline': self.model, 
            'num_features': self.numerical_features,
            'cat_features': self.categorical_features,
            'targets': self.target_columns
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename='model.pkl'):
        data = joblib.load(filename)
        self.model = data['pipeline']
        self.numerical_features = data.get('num_features', self.numerical_features)
        self.categorical_features = data.get('cat_features', self.categorical_features)
        self.target_columns = data.get('targets', self.target_columns)
        print(f"Model loaded from {filename}")
