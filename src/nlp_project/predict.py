from __future__ import annotations
import argparse
from .model import load_model

def main():
    p=argparse.ArgumentParser(description='Predict label for a single text')
    p.add_argument('--model', required=True)
    p.add_argument('--text', required=True)
    a=p.parse_args()
    pipe=load_model(a.model)
    print(pipe.predict([a.text])[0])

if __name__=='__main__':
    main()
