from __future__ import annotations
import re
try:
    import spacy
    _NLP = spacy.blank('xx')
except Exception:
    _NLP=None
from .legacy_import import get_custom_preprocess,get_custom_tokenize
URL_RE = re.compile(r'https?://\S+|www\.\S+')

def default_preprocess(text:str)->str:
    text=text.strip()
    return URL_RE.sub(' URL ', text)

def preprocess(text:str)->str:
    f=get_custom_preprocess()
    if f:
        try: return f(text)
        except Exception: pass
    return default_preprocess(text)

def tokenize(text:str)->list[str]:
    f=get_custom_tokenize()
    if f:
        try: return list(f(text))
        except Exception: pass
    if _NLP: return [t.text for t in _NLP(preprocess(text)) if not t.is_space]
    return preprocess(text).split()

class PreprocessTransformer:
    def fit(self,X,y=None): return self
    def transform(self,X): return [preprocess(x) for x in X]
