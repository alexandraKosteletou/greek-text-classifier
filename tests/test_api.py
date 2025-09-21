import os
import pandas as pd
from fastapi.testclient import TestClient
from nlp_project.model import train_and_eval, TrainConfig

def test_api_predict_smoke(tmp_path):
    df = pd.DataFrame({
        "text": ["καλό", "κακό", "ουδέτερο", "πολύ καλό", "πολύ κακό", "μάλλον ουδέτερο"],
        "label": ["pos", "neg", "neu", "pos", "neg", "neu"],
    })
    mf = tmp_path/'model.joblib'
    
    # Fix 1: Remove the invalid *, * syntax
    train_and_eval(df, TrainConfig(ngram_range=(1,2), max_features=1000), model_path=str(mf))
    
    # Set environment variable first
    os.environ['MODEL_PATH'] = str(mf)
    
    # Fix 2: Import app INSIDE the function AFTER setting env var
    from nlp_project.api import app
    client = TestClient(app)
    
    assert client.get('/health').status_code == 200
    r = client.post('/predict', json={'text':'αυτό ήταν καλό'})
    print(f"Debug - Response: {r.status_code} - {r.text}")  # Add debug line
    assert r.status_code == 200 and 'label' in r.json()