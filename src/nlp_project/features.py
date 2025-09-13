from __future__ import annotations
import re
try:
    import spacy
    _NLP = spacy.blank("xx")
except Exception:
    _NLP = None

from sklearn.base import BaseEstimator, TransformerMixin
from .legacy_import import get_custom_preprocess, get_custom_tokenize

URL_RE = re.compile(r"https?://\\S|www\\.\\S")

def default_preprocess(text: str) -> str:
    text = text.strip()
    text = URL_RE.sub(" URL ", text)
    return text

def preprocess(text: str) -> str:
    custom = get_custom_preprocess()
    if custom:
        try:
            return custom(text)
        except Exception:
            pass
    return default_preprocess(text)

def tokenize(text: str) -> list[str]:
    custom_tok = get_custom_tokenize()
    if custom_tok:
        try:
            return list(custom_tok(text))
        except Exception:
            pass
    if _NLP:
        return [t.text for t in _NLP(preprocess(text)) if not t.is_space]
    # fallback simple split
    return preprocess(text).split()

class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [preprocess(x) for x in X]