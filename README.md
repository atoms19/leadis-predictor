# Learning Disability Risk Predictor

AI-powered system for predicting learning disability risk scores based on child assessment data. Uses PCA dimensionality reduction and Gradient Boosting to predict risk across 8 learning disability categories.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load training data from `training_dat/` folder
- Apply PCA dimensionality reduction by category
- Train Gradient Boosting model with hyperparameter tuning
- Save trained model to `model.pkl`

### 3. Start the API Server

```bash
python server.py
```

Server will start at `http://localhost:5000`

### 4. Test the API

```bash
# In a new terminal
python test_api.py
```

## 📊 Model Architecture

### Algorithm: PCA + Gradient Boosting Regressor with Multi-Output

The model uses a **two-stage approach**:

1. **Stage 1: PCA (Principal Component Analysis) Dimensionality Reduction**
   - Features are organized into 12 categories based on the user-schema
   - PCA is applied separately to each category to reduce dimensionality
   - This preserves category-specific variance while reducing noise

2. **Stage 2: Gradient Boosting Regressor**
   - Uses the PCA-transformed features
   - Wrapped in a Multi-Output Regressor to simultaneously predict 8 risk scores

### Feature Categories & PCA Components

The model organizes features into 12 categories from user-schema.json:

1. **Demographic Info** (3 components)
   - Age, language, schooling, gender
   
2. **Developmental History** (4 components)
   - Multilingual exposure, birth history, speech milestones
   
3. **Sensory Health** (2 components)
   - Hearing and vision status/concerns
   
4. **Family History** (2 components)
   - Family learning difficulties, ADHD history
   
5. **Response Metrics** (3 components)
   - Response accuracy and timing statistics
   
6. **Task Performance** (2 components)
   - Task completion, abandonment, instruction following
   
7. **Attention Metrics** (3 components)
   - Focus duration, attention span, dropoff patterns
   
8. **Memory Metrics** (2 components)
   - Sequence recall and error rates
   
9. **Visual Processing** (2 components)
   - Visual search time, spatial confusion, preferences
   
10. **Auditory Processing** (2 components)
    - Auditory accuracy, replays, preferences
    
11. **Speech Metrics** (2 components)
    - Speech rate and hesitation patterns
    
12. **Reading Metrics** (3 components)
    - Reading speed, accuracy, reversal rates

### Output Predictions

The model predicts risk scores (0-1 range) for 8 learning disability categories:

1. **risk_reading** - Risk of reading difficulties (dyslexia)
2. **risk_writing** - Risk of writing difficulties (dysgraphia)
3. **risk_attention** - Risk of attention disorders (ADHD-like symptoms)
4. **risk_working_memory** - Risk of working memory deficits
5. **risk_expressive_language** - Risk of expressive language difficulties
6. **risk_receptive_language** - Risk of receptive language difficulties
7. **risk_visual_processing** - Risk of visual processing disorders
8. **risk_motor_coordination** - Risk of motor coordination difficulties (dyspraxia)

### Why PCA + Gradient Boosting?

**PCA Benefits:**
- Reduces dimensionality while preserving variance (removes noise)
- Handles multicollinearity between related features
- Category-wise PCA preserves domain knowledge structure
- Improves model generalization

**Gradient Boosting Benefits:**
- Builds multiple decision trees sequentially
- Each tree corrects errors made by previous trees
- Provides excellent predictive accuracy for complex, non-linear relationships
- Robust to overfitting with proper hyperparameter tuning

### Data Processing Pipeline

1. **Feature Extraction**
   - Numerical features: Continuous values (age, scores, timings)
   - Categorical features: Encoded as one-hot vectors

2. **Preprocessing**
   - Missing numerical values: Filled with median
   - Missing categorical values: Filled with 'missing' category
   - Numerical features: Standardized (mean=0, std=1)
   - Categorical features: One-hot encoded

3. **PCA Dimensionality Reduction**
   - Applied separately to each of the 12 feature categories
   - Number of components per category (configurable):
     - Demographic: 3, Developmental: 4, Sensory: 2, Family: 2
     - Response: 3, Task: 2, Attention: 3, Memory: 2
     - Visual: 2, Auditory: 2, Speech: 2, Reading: 3
   - Total components: ~30 (down from 50+ original features)
   - Preserves 85-95% of variance in each category

4. **Model Training**
   - PCA-transformed features fed into Gradient Boosting
   - Hyperparameter tuning via Grid Search CV
   - Multi-output regression for 8 simultaneous predictions

## Evaluation Metrics

### Mean Squared Error (MSE)

**What it measures**: The average squared difference between predicted and actual risk scores.

**Interpretation**:

- Lower is better (0 is perfect)
- If MSE = 0.05, the average prediction error is √0.05 ≈ 0.22 on a 0-1 scale
- Penalizes large errors more heavily due to squaring

**Formula**:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### R² Score (R-squared / Coefficient of Determination)

**What it measures**: How well the model explains the variance in the data.

**Interpretation**:

- Range: -∞ to 1.0 (1.0 is perfect)
- **1.0** = Perfect predictions
- **0.8-0.99** = Excellent model (explains 80-99% of variance)
- **0.6-0.79** = Good model
- **0.4-0.59** = Moderate model
- **0.0-0.39** = Poor model
- **< 0.0** = Model performs worse than simply predicting the mean

**Formula**:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

Where:

- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $\bar{y}$ = mean of actual values

### Per-Target R² Scores

The model reports R² for each of the 8 risk categories separately. This helps identify:

- Which disabilities are easier to predict (higher R²)
- Which may need more features or different modeling approaches (lower R²)

## Hyperparameter Tuning

The model uses Grid Search Cross-Validation to find optimal parameters:

- **n_estimators**: Number of boosting stages (tested: 100)
- **learning_rate**: Step size for each tree's contribution (tested: 0.1)
- **max_depth**: Maximum depth of individual trees (tested: 3)
- **Cross-validation folds**: 3-fold CV for robust evaluation

## Training Process

1. Load all CSV files from `training_data/` folder
2. Split data: 80% training, 20% testing
3. Apply preprocessing pipeline
4. Run Grid Search to find best hyperparameters
5. Train final model on best parameters
6. Evaluate on held-out test set
7. Save model to `model.pkl`

## Making Predictions

Predictions output risk scores between 0.0 and 1.0 for each category:

- **0.0-0.3**: Low risk
- **0.3-0.6**: Moderate risk
- **0.6-1.0**: High risk

Each prediction includes a `sample_id` and all 8 risk scores.

## Usage

### Training the Model

```bash
python train.py
```

### Running Predictions (Command Line)

```bash
python predict.py training_dat/test_data.csv
```

### Testing Model Performance

```bash
python test_pca_model.py
```

Shows comprehensive performance metrics including:
- Overall MSE, RMSE, MAE, R² scores
- Per-target performance metrics
- PCA dimensionality reduction analysis
- Sample predictions vs actual values

---

## 🌐 REST API Server

### Starting the Server

```bash
python server.py
```

The Flask API server provides REST endpoints for making predictions.

### API Endpoints

#### **GET /health**
Health check endpoint

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "model.pkl"
}
```

---

#### **GET /features**
Get list of required features

```bash
curl http://localhost:5000/features
```

**Response:**
```json
{
  "success": true,
  "numerical_features": ["age_months", "mean_response_accuracy", ...],
  "categorical_features": ["primary_language", "gender", ...],
  "target_columns": ["risk_reading", "risk_writing", ...],
  "total_features": 42
}
```

---

#### **GET /model/info**
Get detailed model information including PCA configuration

```bash
curl http://localhost:5000/model/info
```

**Response:**
```json
{
  "success": true,
  "model_path": "model.pkl",
  "pca_transformers": {
    "demographic": {
      "n_components": 3,
      "total_variance_explained": 0.95
    },
    ...
  },
  "numerical_features_count": 35,
  "categorical_features_count": 7
}
```

---

#### **POST /predict**
Make a single prediction

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "attention_dropoff_slope": -0.1,
    "attention_span_average": 200,
    "random_interaction_rate": 0.05,
    "max_sequence_length": 6,
    "sequence_order_error_rate": 0.15,
    "visual_search_time_ms": 3000,
    "left_right_confusion_rate": 0.10,
    "pref_visual": 0.6,
    "auditory_processing_accuracy": 0.85,
    "average_audio_replays": 1,
    "pref_auditory": 0.4,
    "speech_rate_wpm": 100,
    "hesitation_frequency": 5,
    "reading_speed_wpm": 80,
    "reading_accuracy": 0.85,
    "letter_reversal_rate": 0.08,
    "audio_text_mismatch_rate": 0.05
  }'
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "sample_id": "sample_0",
    "targets": {
      "risk_reading": 0.1234,
      "risk_writing": 0.2345,
      "risk_attention": 0.1567,
      "risk_working_memory": 0.2890,
      "risk_expressive_language": 0.1789,
      "risk_receptive_language": 0.1456,
      "risk_visual_processing": 0.2123,
      "risk_motor_coordination": 0.1678
    }
  }
}
```

---

#### **POST /predict/batch**
Make predictions for multiple samples

**Request:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "age_months": 60,
        "primary_language": 0,
        ...
      },
      {
        "age_months": 84,
        "primary_language": 0,
        ...
      }
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "sample_id": "sample_0",
      "targets": { ... }
    },
    {
      "sample_id": "sample_1",
      "targets": { ... }
    }
  ],
  "count": 2
}
```

---

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'age_months': 72,
        'primary_language': 0,
        # ... all other features
    }
)

result = response.json()
if result['success']:
    print(result['prediction']['targets'])
```

---

### JavaScript/Fetch Example

```javascript
// Single prediction
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age_months: 72,
    primary_language: 0,
    // ... all other features
  }),
});

const result = await response.json();
if (result.success) {
  console.log(result.prediction.targets);
}
```

---

## 📝 Feature Reference

See [field-mapping.json](field-mapping.json) for complete feature documentation including:
- Data types (categorical/numerical)
- Value ranges and encodings
- Feature descriptions

### Required Features (42 total)

**Categorical Features (7):**
- `primary_language` - Primary language (0=english)
- `schooling_type` - Educational setting (0-5)
- `gender` - Gender (0=male, 1=female, 2=other, 3=prefer-not-to-say)
- `multilingualExposure` - Multilingual exposure level (0-4)
- `birthHistory` - Birth history (0=full-term, 1=preterm, etc.)
- `hearingStatus` - Hearing status (0-3)
- `visionStatus` - Vision status (0-3)

**Numerical Features (35):**
- Demographics: `age_months`, `multilingual_exposure`
- Developmental: `age_first_word_months`, `age_first_sentence_months`, `history_speech_therapy`, `history_motor_delay`
- Sensory: `hearing_concerns`, `vision_concerns`
- Family: `family_learning_difficulty`, `family_adhd`
- Assessment metrics: Response, task, attention, memory, visual, auditory, speech, and reading metrics

---

## 🎯 Risk Score Interpretation

All risk scores range from 0.0 to 1.0:

- **0.0 - 0.3**: Low risk
- **0.3 - 0.6**: Moderate risk  
- **0.6 - 1.0**: High risk

### Output Categories

1. **risk_reading** - Reading difficulties (dyslexia)
2. **risk_writing** - Writing difficulties (dysgraphia)
3. **risk_attention** - Attention disorders (ADHD)
4. **risk_working_memory** - Working memory deficits
5. **risk_expressive_language** - Expressive language difficulties
6. **risk_receptive_language** - Receptive language difficulties
7. **risk_visual_processing** - Visual processing disorders
8. **risk_motor_coordination** - Motor coordination difficulties (dyspraxia)

---

## 📁 Project Structure

```
leadis-predictor/
├── model.py                    # LDPredictor class with PCA + GBR
├── train.py                    # Training script
├── predict.py                  # Command-line prediction script
├── server.py                   # Flask REST API server
├── test_api.py                 # API testing script
├── test_pca_model.py          # Model performance testing
├── field-mapping.json         # Feature documentation
├── user-schema.json           # User data schema
├── requirements.txt           # Python dependencies
├── model.pkl                  # Trained model (generated)
└── training_dat/
    ├── generate_synthetic_data.py
    ├── training_data.csv
    ├── validation_data.csv
    └── test_data.csv
```

---

## 🔧 Dependencies

- Python 3.8+
- pandas
- scikit-learn
- joblib
- numpy
- flask
- flask-cors
- requests

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📈 Model Performance

Run performance evaluation:
```bash
python test_pca_model.py
```

Expected metrics (on synthetic data):
- **Overall R² Score**: 0.85-0.95 (Excellent)
- **RMSE**: 0.05-0.10
- **Dimensionality Reduction**: ~55 features → 30 PCA components (~45% reduction)

---

## ⚠️ Important Notes

1. **This is a screening tool, not a diagnostic tool** - Risk scores indicate probability, not certainty
2. **Requires all 42 features** - Missing features will cause prediction errors
3. **Categorical features must be encoded** - Use numeric values (0, 1, 2, etc.) as defined in field-mapping.json
4. **Model must be trained first** - Run `train.py` before starting the API server

---

## 🤝 Contributing

1. Generate synthetic training data (if needed):
   ```bash
   cd training_dat
   python generate_synthetic_data.py
   ```

2. Train model with new data:
   ```bash
   python train.py
   ```

3. Test changes:
   ```bash
   python test_pca_model.py
   python test_api.py
   ```

---

## 📄 License

See project license file for details.

---

## 🆘 Troubleshooting

**Server won't start:**
- Ensure model is trained: `python train.py`
- Check port 5000 is available
- Verify all dependencies installed

**Prediction errors:**
- Verify all 42 features are provided
- Check feature encoding matches field-mapping.json
- Ensure numeric values are in valid ranges

**Low accuracy:**
- Retrain with more/better data
- Adjust PCA components per category
- Tune hyperparameters in model.py

---

## Usage (Legacy Command Line)### Prediction

```bash
python predict.py <csv_file>
```

The CSV file should contain the same features used during training.
