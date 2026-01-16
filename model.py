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
from sklearn.decomposition import PCA

class LDPredictor:
    def __init__(self, n_components_per_category=None):
        self.model = None
        self.pca_transformers = {}
        self.n_components_per_category = n_components_per_category or {
            'demographic': 3,
            'developmental': 4,
            'sensory': 2,
            'family': 2,
            'response': 3,
            'task': 2,
            'attention': 3,
            'memory': 2,
            'visual': 2,
            'auditory': 2,
            'speech': 2,
            'reading': 3
        }
        
        # Targets
        self.target_columns = [
            'risk_reading', 'risk_writing', 'risk_attention', 
            'risk_working_memory', 'risk_expressive_language', 
            'risk_receptive_language', 'risk_visual_processing', 
            'risk_motor_coordination'
        ]
        
        # Feature groups based on user-schema.json categories
        # 1. Demographic Info (numerical + categorical)
        self.demographic_numerical = ['age_months']
        self.demographic_categorical = ['primary_language', 'schooling_type', 'gender']
        
        # 2. Developmental History
        self.developmental_numerical = [
            'multilingual_exposure', 'age_first_word_months', 
            'age_first_sentence_months', 'history_speech_therapy', 
            'history_motor_delay'
        ]
        self.developmental_categorical = ['multilingualExposure', 'birthHistory']
        
        # 3. Sensory Health
        self.sensory_numerical = ['hearing_concerns', 'vision_concerns']
        self.sensory_categorical = ['hearingStatus', 'visionStatus']
        
        # 4. Family History
        self.family_numerical = ['family_learning_difficulty']
        self.family_categorical = ['family_adhd']
        
        # 5. Assessment Metrics - Response Metrics
        self.response_numerical = [
            'mean_response_accuracy', 'response_accuracy_std', 
            'mean_response_time_ms', 'response_time_std_ms'
        ]
        
        # 6. Assessment Metrics - Task Performance
        self.task_numerical = [
            'task_completion_rate', 'task_abandonment_count',
            'instruction_follow_accuracy'
        ]
        
        # 7. Assessment Metrics - Attention Metrics
        self.attention_numerical = [
            'mean_focus_duration_sec', 'attention_dropoff_slope',
            'attention_span_average', 'random_interaction_rate'
        ]
        
        # 8. Assessment Metrics - Memory Metrics
        self.memory_numerical = [
            'max_sequence_length', 'sequence_order_error_rate'
        ]
        
        # 9. Assessment Metrics - Visual Processing
        self.visual_numerical = [
            'visual_search_time_ms', 'left_right_confusion_rate', 'pref_visual'
        ]
        
        # 10. Assessment Metrics - Auditory Processing
        self.auditory_numerical = [
            'auditory_processing_accuracy', 'average_audio_replays', 'pref_auditory'
        ]
        
        # 11. Assessment Metrics - Speech Metrics
        self.speech_numerical = ['speech_rate_wpm', 'hesitation_frequency']
        
        # 12. Assessment Metrics - Reading Metrics
        self.reading_numerical = [
            'reading_speed_wpm', 'reading_accuracy',
            'letter_reversal_rate', 'audio_text_mismatch_rate'
        ]
        
        # All categorical and numerical features
        self.categorical_features = (
            self.demographic_categorical + self.developmental_categorical +
            self.sensory_categorical + self.family_categorical
        )
        
        self.numerical_features = (
            self.demographic_numerical + self.developmental_numerical +
            self.sensory_numerical + self.family_numerical +
            self.response_numerical + self.task_numerical +
            self.attention_numerical + self.memory_numerical +
            self.visual_numerical + self.auditory_numerical +
            self.speech_numerical + self.reading_numerical
        )

    def _build_preprocessing_pipeline(self):
        """Build preprocessing pipeline for categorical and numerical features."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical features are already encoded as integers in the data
        # Use -1 as fill value which will be treated as a new category by OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.float64))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor

    def _apply_pca_by_category(self, X_preprocessed, fit=True):
        """
        Apply PCA to each feature category separately.
        
        Args:
            X_preprocessed: Preprocessed feature array from ColumnTransformer
            fit: If True, fit new PCA transformers. If False, use existing ones.
        
        Returns:
            Array with PCA-transformed features concatenated
        """
        # Get feature names after preprocessing
        preprocessor = self.preprocessor
        
        # Calculate feature positions after preprocessing
        num_features = len(self.numerical_features)
        
        # Get encoded categorical features count
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer.named_steps['onehot'], 'categories_'):
            num_cat_features = sum(len(cats) for cats in cat_transformer.named_steps['onehot'].categories_)
        else:
            num_cat_features = 0
        
        # Split preprocessed data back into categories
        # Numerical features come first, then categorical (one-hot encoded)
        numerical_encoded = X_preprocessed[:, :num_features]
        categorical_encoded = X_preprocessed[:, num_features:]
        
        # Define category groups with their indices in the numerical array
        category_groups = {
            'demographic': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.demographic_numerical],
                'cat_start': 0,  # Will calculate below
                'cat_count': 0   # Will calculate below
            },
            'developmental': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.developmental_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'sensory': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.sensory_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'family': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.family_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'response': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.response_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'task': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.task_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'attention': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.attention_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'memory': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.memory_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'visual': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.visual_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'auditory': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.auditory_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'speech': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.speech_numerical],
                'cat_start': 0,
                'cat_count': 0
            },
            'reading': {
                'num_indices': [i for i, f in enumerate(self.numerical_features) if f in self.reading_numerical],
                'cat_start': 0,
                'cat_count': 0
            }
        }
        
        # Calculate categorical feature positions (after one-hot encoding)
        if num_cat_features > 0:
            cat_groups = [
                self.demographic_categorical,
                self.developmental_categorical,
                self.sensory_categorical,
                self.family_categorical
            ]
            group_names = ['demographic', 'developmental', 'sensory', 'family']
            
            cat_position = 0
            for group_name, cat_features in zip(group_names, cat_groups):
                if len(cat_features) > 0:
                    # Count one-hot encoded features for this group
                    cat_count = 0
                    for feat in cat_features:
                        feat_idx = self.categorical_features.index(feat)
                        cat_count += len(cat_transformer.named_steps['onehot'].categories_[feat_idx])
                    
                    category_groups[group_name]['cat_start'] = cat_position
                    category_groups[group_name]['cat_count'] = cat_count
                    cat_position += cat_count
        
        # Apply PCA to each category
        pca_components = []
        
        for category_name, group_info in category_groups.items():
            # Combine numerical and categorical features for this category
            num_indices = group_info['num_indices']
            cat_start = group_info['cat_start']
            cat_count = group_info['cat_count']
            
            category_features = []
            
            # Add numerical features
            if len(num_indices) > 0:
                category_features.append(numerical_encoded[:, num_indices])
            
            # Add categorical features
            if cat_count > 0:
                cat_end = cat_start + cat_count
                category_features.append(categorical_encoded[:, cat_start:cat_end])
            
            # Skip if no features in this category
            if len(category_features) == 0:
                continue
            
            # Combine features
            X_category = np.hstack(category_features)
            
            # Apply PCA
            n_components = min(
                self.n_components_per_category[category_name],
                X_category.shape[1],
                X_category.shape[0]  # Can't have more components than samples
            )
            
            if fit:
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X_category)
                self.pca_transformers[category_name] = pca
            else:
                pca = self.pca_transformers[category_name]
                X_pca = pca.transform(X_category)
            
            pca_components.append(X_pca)
        
        # Concatenate all PCA components
        X_final = np.hstack(pca_components)
        
        return X_final

    def train(self, df):
        """
        Trains the model using PCA for dimensionality reduction and GridSearchCV for hyperparameter tuning.
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

        # Build preprocessing pipeline
        print("Building preprocessing pipeline...")
        self.preprocessor = self._build_preprocessing_pipeline()
        
        # Fit preprocessor and transform training data
        print("Applying preprocessing...")
        X_train_preprocessed = self.preprocessor.fit_transform(X_train)
        X_test_preprocessed = self.preprocessor.transform(X_test)
        
        # Apply PCA by category
        print("Applying PCA dimensionality reduction by category...")
        X_train_pca = self._apply_pca_by_category(X_train_preprocessed, fit=True)
        X_test_pca = self._apply_pca_by_category(X_test_preprocessed, fit=False)
        
        print(f"Original features: {X_train_preprocessed.shape[1]}")
        print(f"PCA-reduced features: {X_train_pca.shape[1]}")
        print("\nPCA components per category:")
        for cat_name, pca in self.pca_transformers.items():
            print(f"  {cat_name}: {pca.n_components_} components (explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 4]
        }

        print("\nStarting Grid Search for hyperparameter tuning...")
        base_estimator = GradientBoostingRegressor(random_state=42)
        regressor = MultiOutputRegressor(base_estimator)
        
        grid_search = GridSearchCV(
            regressor, 
            param_grid={'estimator__' + k: v for k, v in param_grid.items()},
            cv=3, 
            n_jobs=-1, 
            scoring='r2', 
            verbose=1
        )
        
        grid_search.fit(X_train_pca, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"\nBest Parameters: {grid_search.best_params_}")

        # Evaluate
        print("\n--- Model Evaluation ---")
        y_pred = self.model.predict(X_test_pca)
        
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
        
        # Preprocess
        X_preprocessed = self.preprocessor.transform(X_new)
        
        # Apply PCA transformations
        X_pca = self._apply_pca_by_category(X_preprocessed, fit=False)
        
        # Predict
        predictions = self.model.predict(X_pca)
        
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
            'model': self.model,
            'preprocessor': self.preprocessor,
            'pca_transformers': self.pca_transformers,
            'num_features': self.numerical_features,
            'cat_features': self.categorical_features,
            'targets': self.target_columns,
            'n_components': self.n_components_per_category
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename='model.pkl'):
        data = joblib.load(filename)
        self.model = data['model']
        self.preprocessor = data['preprocessor']
        self.pca_transformers = data['pca_transformers']
        self.numerical_features = data.get('num_features', self.numerical_features)
        self.categorical_features = data.get('cat_features', self.categorical_features)
        self.target_columns = data.get('targets', self.target_columns)
        self.n_components_per_category = data.get('n_components', self.n_components_per_category)
        print(f"Model loaded from {filename}")
