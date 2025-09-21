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
def some_function():
    global model, loaded_path  # Remove the * and add proper indentation
    # Your function code here
    path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    _loaded_path = path
    try:
        _model = load_model(path)
    except Exception as e:
        # Μην ρίχνεις το app· άφησε το /health να δείξει ότι δεν φορτώθηκε
        print(f"[startup] Failed to load model from {path}: {e}")
        _model = None
 

@app.get('/health')
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": _loaded_path}

@app.post('/predict', response_model=PredictOut)
def predict(inp:PredictIn):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label = _model.predict([inp.text])[0]  # type: ignore
    return PredictOut(label=label)