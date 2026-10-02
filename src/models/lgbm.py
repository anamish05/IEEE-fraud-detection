import gc
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def train_time_series_lgb(
        df: pd.DataFrame,
        features: List[str],
        target: str,
        params: Dict[str,Any],
        n_splits: int=5,
        num_boost_round=1500,
        early_stopping_rounds: int=50
) -> Tuple[List[lgb.Booster], np.ndarray, List[float]]:
    
    """Trains LightGBM models using TimeSeriesSplit cross-validation."""

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_preds = np.full(len(df), np.nan, dtype=np.float32)
    fold_scores = []
    models = []

    print(f"Starting {n_splits}-Fold TimeSeriesSplit across full dataset...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print(f"{'='*22} Fold {fold} / {n_splits} {'='*22}")
        print(f"Train window: indices {train_idx[0]:>7,} -> {train_idx[-1]:>7,} (Count: {len(train_idx):,})")
        print(f"Val window:   indices {val_idx[0]:>7,} -> {val_idx[-1]:>7,} (Count: {len(val_idx):,})")

        # Slice strictly by index array (zero full-table duplication)
        X_tr = df.iloc[train_idx][features]
        y_tr = df.iloc[train_idx][target]
        
        X_val = df.iloc[val_idx][features]
        y_val = df.iloc[val_idx][target]

        # free_raw_data=True releases memory once internal histograms are formed
        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

        del X_tr, y_tr
        gc.collect()

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )

        # Predict validation segment
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        val_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(val_auc)
        models.append(model)

        print(f"Fold {fold} Best Iteration: {model.best_iteration} | Val ROC-AUC: {val_auc:.5f}\n")

        # Clean up before allocating the next fold
        del dtrain, dval, X_val, y_val
        gc.collect()

    return models, oof_preds, fold_scores
