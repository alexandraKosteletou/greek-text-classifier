from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import os
from .model import load_model

MODEL_PATH=os.getenv('MODEL_PATH','artifacts/model.joblib')

class PredictIn(BaseModel):
    text:str
class PredictOut(BaseModel):
    label:str

app=FastAPI(title='Greek Text Classifier API')
_model=None

@app.on_event('startup')
def _load():
    global _model
    _model=load_model(MODEL_PATH)

@app.get('/health')
def health():
    return {'status':'ok','model_loaded': _model is not None}

@app.post('/predict', response_model=PredictOut)
def predict(inp:PredictIn):
    return PredictOut(label=_model.predict([inp.text])[0])
