# Learning Disability Risk Predictor

This project uses machine learning to predict risk scores for various learning disabilities based on child assessment data.

## Model Architecture

### Algorithm: Gradient Boosting Regressor with Multi-Output

The model uses a **Gradient Boosting Regressor** wrapped in a **Multi-Output Regressor** to simultaneously predict risk scores for 8 different learning disability categories:

1. **risk_reading** - Risk of reading difficulties
2. **risk_writing** - Risk of writing difficulties
3. **risk_attention** - Risk of attention disorders (ADHD-like symptoms)
4. **risk_working_memory** - Risk of working memory deficits
5. **risk_expressive_language** - Risk of expressive language difficulties
6. **risk_receptive_language** - Risk of receptive language difficulties
7. **risk_visual_processing** - Risk of visual processing disorders
8. **risk_motor_coordination** - Risk of motor coordination difficulties

### Why Gradient Boosting?

Gradient Boosting is an ensemble learning method that:

- Builds multiple decision trees sequentially
- Each tree corrects errors made by previous trees
- Provides excellent predictive accuracy for complex, non-linear relationships
- Handles both categorical and numerical features effectively

### Data Pipeline

The model includes a complete preprocessing pipeline:

1. **Numerical Features** (50+ features):

   - **Demographics**: age_months, multilingual_exposure
   - **Development History**: age_first_word_months, age_first_sentence_months
   - **Medical History**: history_speech_therapy, vision_concerns, hearing_concerns, family_learning_difficulty
   - **Assessment Metrics**:
     - Response accuracy and timing
     - Memory recall scores
     - Visual discrimination accuracy
     - Language fluency metrics
     - Reading performance indicators
2. **Categorical Features**:

   - primary_language
   - schooling_type
   - gender
3. **Preprocessing Steps**:

   - Missing numerical values filled with median
   - Missing categorical values filled with 'missing'
   - Numerical features standardized (mean=0, std=1)
   - Categorical features one-hot encoded

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

### Training

```bash
python train.py
```

### Prediction

```bash
python predict.py <csv_file>
```

The CSV file should contain the same features used during training.
