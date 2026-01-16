import pandas as pd
import joblib
from ld_prediction_model import LD_EarlyDetectionModel, generate_dummy_data_folder

# 1. Load the System
system = LD_EarlyDetectionModel()
try:
    system.load_model('ld_model.pkl') # Make sure this file exists from previous training
except:
    print("No model found! Please run the training script first.")
    exit()

# 2. Point to a new CSV file
# (Creating a dummy one here just so the script runs for you)
dummy_csv = 'new_patients.csv'
generate_dummy_data_folder('temp_data') # reusing helper to generate data
# Just grabbing one file to simulate a "new upload"
import shutil
shutil.copy('temp_data/clinic_north.csv', dummy_csv) 

# 3. Run Prediction
print(f"Analyzing {dummy_csv}...")
results = system.predict_analysis(dummy_csv)

# 4. Print the Dictionary Results (Simulating an API Response)
import json
print(json.dumps(results, indent=2))
