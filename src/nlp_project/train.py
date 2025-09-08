from __future__ import annotations
import argparse
from .data import load_csv
from .model import TrainConfig, train_and_eval

def main():
    p=argparse.ArgumentParser(description='Train Greek text classifier')
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='artifacts/model.joblib')
    p.add_argument('--ngrams', type=int, nargs=2, default=(1,2))
    p.add_argument('--max_features', type=int, default=50000)
    p.add_argument('--C', type=float, default=1.0)
    args=p.parse_args()
    df=load_csv(args.data)
    cfg=TrainConfig(ngram_range=tuple(args.ngrams), max_features=args.max_features, C=args.C)
    _, rep = train_and_eval(df, cfg, model_path=args.model)
    print(rep)

if __name__=='__main__':
    main()
