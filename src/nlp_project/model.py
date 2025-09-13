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
    ngram_range: tuple[int, int] = (1,2)
    max_features: Optional[int] = 50000
    C: float = 1.0

def build_pipeline(cfg: TrainConfig) -> Pipeline:
    custom_vec_factory = get_custom_vectorizer()
    if custom_vec_factory:
        try:
            vectorizer = custom_vec_factory()
        except Exception:
            vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=cfg.ngram_range,
                                         max_features=cfg.max_features)
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=cfg.ngram_range,
                                     max_features=cfg.max_features)
    clf = LinearSVC(C=cfg.C)
    pipe = Pipeline([
        ("prep", PreprocessTransformer()),
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe

def train_and_eval(df, cfg: TrainConfig, model_path: str | None = None):
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import math

    X = df["text"].tolist()
    y = df["label"].tolist()
    n = len(y)
    k = len(set(y))
    counts = Counter(y)

    # stratify μόνο αν "χωράει" για όλες τις κλάσεις, αλλιώς fallback
    test_size_float = max(0.2, (k / n)  1e-9) if n > 0 else 0.2
    can_stratify = (
        n >= 2 and all(c >= 2 for c in counts.values())
        and math.ceil(test_size_float * n) >= k
    )
    try:
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size_float, random_state=42, stratify=y
            )
        else:
            raise ValueError("Tiny dataset — skip stratify")
    except ValueError:
        test_size_int = max(1, int(round(0.2 * n)))
        test_size_int = min(test_size_int, n - 1) if n > 1 else 1
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