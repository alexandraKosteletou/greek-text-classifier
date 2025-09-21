from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from .model import load_model

DEFAULT_MODEL_PATH = "artifacts/model.joblib"

class PredictIn(BaseModel):
    text:str
class PredictOut(BaseModel):
    label:str

app=FastAPI(title='Greek Text Classifier API')
_model=None
_loaded_path = None

@app.on_event('startup')
async def load_model_on_startup():
    global model, loaded_path
    try:
        model_path = os.getenv('MODEL_PATH', 'artifacts/model.joblib')
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
    return {"status": "ok", "model_loaded": _model is not None, "model_path": _loaded_path}

@app.post("/predict")
async def predict(request: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available - please check model loading")
    
    try:
        prediction = model.predict([request.text])
        return {"label": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    r = client.post('/predict', json={'text':'αυτό ήταν καλό'})
    print(f"Debug - Response: {r.status_code} - {r.text}")  # Add this line
    assert r.status_code == 200 and 'label' in r.json()