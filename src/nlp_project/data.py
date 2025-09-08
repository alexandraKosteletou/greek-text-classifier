from __future__ import annotations
import pandas as pd

def load_csv(path: str, text_col='text', label_col='label'):
    df = pd.read_csv(path)
    if text_col not in df or label_col not in df:
        raise ValueError(f"CSV must have columns '{text_col}' and '{label_col}'")
    return df[[text_col,label_col]].dropna().reset_index(drop=True)
