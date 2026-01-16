import os
import glob
import pandas as pd
from model import LDPredictor

def load_training_data(data_folder='training_data'):
    print(f"Looking for data in folder: '{data_folder}'...")
    search_path = os.path.join(data_folder, "*.csv")
    all_files = glob.glob(search_path)
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}'!")
        
    print(f"Found {len(all_files)} files: {[os.path.basename(f) for f in all_files]}")

    # Define required targets to validate data
    required_targets = [
        'risk_reading', 'risk_writing', 'risk_attention', 
        'risk_working_memory', 'risk_expressive_language', 
        'risk_receptive_language', 'risk_visual_processing', 
        'risk_motor_coordination'
    ]

    df_list = []
    for filename in all_files:
        try:
            temp_df = pd.read_csv(filename)
            # Check if all required targets are present
            if all(target in temp_df.columns for target in required_targets):
                df_list.append(temp_df)
            else:
                print(f"Skipping {filename}: Missing target columns.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        raise ValueError("No valid training data could be loaded.")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset size: {df.shape[0]} rows.")
    return df

if __name__ == "__main__":
    try:
        # 1. Load Data
        df = load_training_data()
        
        # 2. Initialize and Train
        predictor = LDPredictor()
        predictor.train(df)
        
        # 3. Save Model
        predictor.save('model.pkl')
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
