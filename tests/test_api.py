import os
import pandas as pd
from fastapi.testclient import TestClient
# Remove this line: from nlp_project.api import app
from nlp_project.model import train_and_eval, TrainConfig

def test_api_predict_smoke(tmp_path):
    # Create model first
    df = pd.DataFrame({...})
    mf = tmp_path/'model.joblib'
    _ = train_and_eval(df, TrainConfig(ngram_range=(1,2), max_features=1000), model_path=str(mf))
    
    # Set env var BEFORE importing app
    os.environ['MODEL_PATH'] = str(mf)
    
    # Import app AFTER env var is set
    from nlp_project.api import app
    client = TestClient(app)
    
    # Rest of test...
def test_api_predict_smoke(tmp_path):
    df = pd.DataFrame({
    "text": ["καλό", "κακό", "ουδέτερο", "πολύ καλό", "πολύ κακό", "μάλλον ουδέτερο"],
    "label": ["pos",  "neg",  "neu",      "pos",       "neg",       "neu"],})

    mf=tmp_path/'model.joblib'
    _, _ = train_and_eval(df, TrainConfig(ngram_range=(1,2), max_features=1000), model_path=str(mf))
    os.environ['MODEL_PATH']=str(mf)
    client=TestClient(app)
    assert client.get('/health').status_code==200
    r=client.post('/predict', json={'text':'αυτό ήταν καλό'})
    assert r.status_code==200 and 'label' in r.json()
