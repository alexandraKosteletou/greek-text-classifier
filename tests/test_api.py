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
    train_and_eval(df, TrainConfig(ngram_range=(1,2), max_features=1000), model_path=str(mf))
    
    # Set environment variable
    os.environ['MODEL_PATH'] = str(mf)
    
    from nlp_project.api import app
    client = TestClient(app)
    
    # Reload the model after setting env var
    reload_response = client.post('/reload_model')
    print(f"Reload response: {reload_response.json()}")
    
    assert client.get('/health').status_code == 200
    r = client.post('/predict', json={'text':'αυτό ήταν καλό'})
    print(f"Debug - Response: {r.status_code} - {r.text}")
    assert r.status_code == 200 and 'label' in r.json()
    
    # Debug: Check what endpoints are available
    print(f"Available routes: {[route.path for route in app.routes]}")
    
    assert client.get('/health').status_code == 200
    r = client.post('/predict', json={'text':'αυτό ήταν καλό'})