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

def train_and_eval(df, cfg: TrainConfig, model_path: str | None = None):
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import math
 
    X = df["text"].tolist()
    y = df["label"].tolist()
    n = len(y)
    k = len(set(y))
    counts = Counter(y)
 
     # Προσπάθησε με stratify, με test_size που να "χωράει" όλες τις κλάσεις στο test
    test_size_float = max(0.2, (k / n)  1e-9)  # π.χ. για n=5,k=3 ⇒ 0.6 ⇒ ceil(3)
    can_stratify = all(c >= 2 for c in counts.values()) and math.ceil(test_size_float * n) >= k
 
    try:
        if can_stratify:
           X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size_float, random_state=42, stratify=y
             )
        else:
           raise ValueError("Tiny dataset — skip stratify")
    except ValueError:
 # Fallback: χωρίς stratify, με τουλάχιστον 1 δείγμα στο test
       test_size_int = max(1, int(round(0.2 * n)))
       test_size_int = min(test_size_int, n - 1)
       X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=test_size_int, random_state=42, stratify=None
       )
 
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=False, digits=3)
    if model_path:
       joblib.dump(pipe, model_path)
    return pipe, report

def load_model(path:str)->Pipeline:
   return joblib.load(path)
