"""
Test script to evaluate the pre-trained PCA-based model and show performance metrics.
"""
import pandas as pd
import numpy as np
from model import LDPredictor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def test_model(model_path='model.pkl'):
    print("=" * 80)
    print(" " * 20 + "PCA-Based Learning Disability Predictor")
    print(" " * 30 + "Model Performance Report")
    print("=" * 80)
    
    # 1. Load pre-trained model
    print("\n[1] LOADING PRE-TRAINED MODEL")
    print("-" * 80)
    
    if not os.path.exists(model_path):
        print(f"   ✗ Model file '{model_path}' not found!")
        print(f"   Please train the model first using: python train.py")
        return
    
    predictor = LDPredictor()
    predictor.load(model_path)
    
    print(f"   ✓ Model loaded from: {model_path}")
    print(f"   Features: {len(predictor.numerical_features)} numerical + {len(predictor.categorical_features)} categorical")
    
    # 2. Load test data
    print("\n[2] LOADING TEST DATA")
    print("-" * 80)
    test_df = pd.read_csv('training_dat/test_data.csv')
    print(f"   Test samples: {len(test_df):,}")
    
    # 3. Evaluate on test set
    print("\n[3] TEST SET EVALUATION")
    print("-" * 80)
    
    target_columns = predictor.target_columns
    X_test = test_df[predictor.numerical_features + predictor.categorical_features]
    y_test = test_df[target_columns]
    
    # Preprocess and predict
    X_test_preprocessed = predictor.preprocessor.transform(X_test)
    X_test_pca = predictor._apply_pca_by_category(X_test_preprocessed, fit=False)
    y_pred = predictor.model.predict(X_test_pca)
    
    # Overall metrics
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_r2 = r2_score(y_test, y_pred)
    
    print(f"\n   Overall Performance:")
    print(f"      Mean Squared Error (MSE):  {overall_mse:.6f}")
    print(f"      Root Mean Squared Error:   {overall_rmse:.6f}")
    print(f"      Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"      R² Score:                  {overall_r2:.6f}")
    
    # Per-target metrics
    print(f"\n   Per-Target Performance:")
    print(f"   {'Target':<30} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
    print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    
    target_metrics = []
    for i, target in enumerate(target_columns):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        target_metrics.append({'target': target, 'r2': r2, 'rmse': rmse, 'mae': mae})
        print(f"   {target:<30} {r2:<12.6f} {rmse:<12.6f} {mae:<12.6f}")
    
    # 4. PCA Analysis
    print(f"\n[4] PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("-" * 80)
    print(f"\n   Original feature space: {X_test_preprocessed.shape[1]} dimensions")
    print(f"   Reduced feature space:  {X_test_pca.shape[1]} dimensions")
    print(f"   Compression ratio:      {(1 - X_test_pca.shape[1]/X_test_preprocessed.shape[1])*100:.1f}% reduction")
    
    print(f"\n   PCA Components by Category:")
    print(f"   {'Category':<25} {'Components':<12} {'Variance Explained':<20}")
    print(f"   {'-'*25} {'-'*12} {'-'*20}")
    
    for cat_name, pca in predictor.pca_transformers.items():
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"   {cat_name:<25} {pca.n_components_:<12} {variance_explained:<20.4f}")
    
    # 5. Prediction examples
    print(f"\n[5] SAMPLE PREDICTIONS")
    print("-" * 80)
    
    # Show 3 sample predictions vs actual
    sample_indices = [0, len(test_df)//2, len(test_df)-1]
    
    for idx in sample_indices:
        print(f"\n   Sample #{idx + 1}:")
        print(f"   {'Target':<30} {'Actual':<12} {'Predicted':<12} {'Difference':<12}")
        print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        
        for i, target in enumerate(target_columns):
            actual = y_test.iloc[idx, i]
            predicted = y_pred[idx, i]
            diff = abs(actual - predicted)
            print(f"   {target:<30} {actual:<12.4f} {predicted:<12.4f} {diff:<12.4f}")
    
    # 6. Save model
    print(f"\n[6] MODEL INFORMATION")
    print("-" * 80)
    print(f"   Model file: {model_path}")
    print(f"   ✓ Model is ready for predictions")
    
    # 7. Test prediction API
    print(f"\n[7] TESTING PREDICTION API")
    print("-" * 80)
    
    # Make prediction with loaded model
    test_sample = test_df.head(1)
    predictions = predictor.predict(test_sample)
    
    print(f"   Sample prediction output:")
    for pred in predictions:
        print(f"\n   Sample ID: {pred['sample_id']}")
        print(f"   Risk Scores:")
        for risk_name, risk_value in pred['targets'].items():
            print(f"      {risk_name}: {risk_value:.4f}")
    
    print(f"\n   ✓ Prediction API working correctly")
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"\n   Overall R² Score:        {overall_r2:.4f} ({'Excellent' if overall_r2 > 0.9 else 'Good' if overall_r2 > 0.7 else 'Moderate' if overall_r2 > 0.5 else 'Poor'})")
    print(f"   Overall RMSE:            {overall_rmse:.4f}")
    print(f"   Dimensionality Reduction: {X_test_preprocessed.shape[1]} → {X_test_pca.shape[1]} features")
    
    best_target = max(target_metrics, key=lambda x: x['r2'])
    worst_target = min(target_metrics, key=lambda x: x['r2'])
    
    print(f"\n   Best Performing Target:  {best_target['target']} (R²={best_target['r2']:.4f})")
    print(f"   Worst Performing Target: {worst_target['target']} (R²={worst_target['r2']:.4f})")
    
    print("\n" + "=" * 80)
    print(" " * 32 + "✓ TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model.pkl'
    test_model(model_path)
