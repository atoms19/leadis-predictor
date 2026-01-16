"""
Flask API Server for Learning Disability Risk Prediction
Privacy-Preserving Edition
"""
from flask import Flask
from flask_cors import CORS
from app.database import engine, Base
from app.routes import api_bp

# Initialize Database Tables
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app)

# Register the API Blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    print("=" * 60)
    print("Learning Disability Risk Prediction API Server")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health                 - Health check")
    print("  POST /session/create         - Create new quiz session")
    print("  POST /predict                - Make prediction & save data")
    print("  GET  /session/<credential>   - Retrieve session data")
    print("\n" + "=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
