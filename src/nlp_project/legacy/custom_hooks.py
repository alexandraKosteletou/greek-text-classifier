from __future__ import annotations
import re

def custom_tokenize(text:str):
    import re
    return [t for t in re.split(r'\W+', text) if t]

def custom_preprocess(text:str):
    return ' '.join(custom_tokenize(text.lower()))

def custom_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1,2), max_features=50000)
