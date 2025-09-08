from __future__ import annotations
import importlib

def _try_import(name:str):
    try: return importlib.import_module(name)
    except Exception: return None

def load_custom_hooks():
    hooks={}
    mod=_try_import('nlp_project.legacy.custom_hooks')
    if not mod: return hooks
    for a in ('custom_preprocess','custom_tokenize','custom_vectorizer'):
        f=getattr(mod,a,None)
        if f: hooks[a]=f
    return hooks

def get_custom_preprocess(): return load_custom_hooks().get('custom_preprocess')

def get_custom_tokenize(): return load_custom_hooks().get('custom_tokenize')

def get_custom_vectorizer(): return load_custom_hooks().get('custom_vectorizer')
