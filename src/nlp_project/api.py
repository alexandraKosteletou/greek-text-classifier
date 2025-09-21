from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib  # Add this missing import
from .model import load_model

DEFAULT_MODEL_PATH = "artifacts/model.joblib"

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str

app = FastAPI(title='Greek Text Classifier API')

# Fix variable names - no asterisks allowed
model = None
loaded_path = None

@app.on_event('startup')
async def load_model_on_startup():
    global model, loaded_path  # Use consistent variable names
    try:
        model_path = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
        print(f"Trying to load model from: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            loaded_path = model_path
            print(f"✅ Model loaded successfully")
        else:
            print(f"❌ Model file not found")
            model = None
            loaded_path = None
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None
        loaded_path = None
 
@app.get('/health')
def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None, 
        "model_path": loaded_path
    }

@app.post("/reload_model")
async def reload_model():
    global model, loaded_path
    try:
        model_path = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
        print(f"Reloading model from: {model_path}")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            loaded_path = model_path
            return {"status": "success", "model_path": model_path}
        else:
            model = None
            loaded_path = None
            return {"status": "failed", "error": f"Model not found at {model_path}"}
    except Exception as e:
        model = None
        loaded_path = None
        return {"status": "failed", "error": str(e)}