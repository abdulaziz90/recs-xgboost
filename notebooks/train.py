import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 1) Read everything from the single “train” channel
    train_dir = "/opt/ml/input/data/train"
    X = pd.read_parquet(os.path.join(train_dir, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(train_dir, "y_train.parquet")).squeeze()

    # 2) Log‑transform target
    y = np.log1p(y)

    # 3) Split out your own validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Wrap in DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    # 5) Parse hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective",            type=str,   default="reg:squarederror")
    parser.add_argument("--max_depth",            type=int,   default=6)
    parser.add_argument("--eta",                  type=float, default=0.1)
    parser.add_argument("--subsample",            type=float, default=0.8)
    parser.add_argument("--colsample_bytree",     type=float, default=0.8)
    parser.add_argument("--num_round",            type=int,   default=100)
    parser.add_argument("--early_stopping_rounds",type=int,   default=25)
    args = parser.parse_args()

    params = {
        "objective":        args.objective,
        "max_depth":        args.max_depth,
        "eta":              args.eta,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    # 6) Train with validation
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )

    # 7) Save model for SageMaker
    model_dir = os.environ["SM_MODEL_DIR"]
    model.save_model(os.path.join(model_dir, "model.bst"))
