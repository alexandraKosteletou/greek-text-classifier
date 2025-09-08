from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from .features import tokenize, PreprocessTransformer
from .legacy_import import get_custom_vectorizer

@dataclass
class TrainConfig:
    ngram_range: tuple[int,int]=(1,2)
    max_features: Optional[int]=50000
    C: float = 1.0

def build_pipeline(cfg:TrainConfig)->Pipeline:
    factory=get_custom_vectorizer()
    if factory:
        try:
            vec=factory()
        except Exception:
            vec=TfidfVectorizer(tokenizer=tokenize, ngram_range=cfg.ngram_range, max_features=cfg.max_features)
    else:
        vec=TfidfVectorizer(tokenizer=tokenize, ngram_range=cfg.ngram_range, max_features=cfg.max_features)
    clf=LinearSVC(C=cfg.C)
    return Pipeline([('prep',PreprocessTransformer()),('tfidf',vec),('clf',clf)])

def train_and_eval(df,cfg:TrainConfig,model_path:str|None=None):
    from sklearn.model_selection import train_test_split
    Xtr,Xv,ytr,yv = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'])
    pipe=build_pipeline(cfg)
    pipe.fit(Xtr,ytr)
    yhat=pipe.predict(Xv)
    report=classification_report(yv,yhat,output_dict=False,digits=3)
    if model_path: joblib.dump(pipe, model_path)
    return pipe, report

def load_model(path:str)->Pipeline:
    return joblib.load(path)
