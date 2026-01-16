import sys
import pandas as pd
import json
from model import LDPredictor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    
    try:
        # 1. Load Model
        predictor = LDPredictor()
        predictor.load('model.pkl')
        
        # 2. Load Data
        df_new = pd.read_csv(csv_file)
        
        # 3. Predict
        results = predictor.predict(df_new)
        
        # 4. Output JSON
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
