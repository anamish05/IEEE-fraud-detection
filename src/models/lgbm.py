import gc
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.lightgbm


def train_time_series_lgb(
        df: pd.DataFrame,
        features: List[str],
        target: str,
        params: Dict[str,Any],
        n_splits: int=5,
        num_boost_round=1500,
        early_stopping_rounds: int=50,
        thresh: float=0.2,
        log_mlflow=True
) -> Tuple[List[lgb.Booster], np.ndarray, Dict[str, List[float]]]:
    
    """Trains LightGBM models using TimeSeriesSplit cross-validation."""

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_preds = np.full(len(df), np.nan, dtype=np.float32)
    metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "precision": [],
        "recall": []
    }
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
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds
        
        val_roc_auc = roc_auc_score(y_val, val_preds)
        val_pr_auc = average_precision_score(y_val, val_preds)
        val_precision = precision_score(y_val, (val_preds >= thresh).astype(int), zero_division=0)
        val_recall = recall_score(y_val, (val_preds >= thresh).astype(int), zero_division=0)

        metrics["roc_auc"].append(val_roc_auc)
        metrics["pr_auc"].append(val_pr_auc)
        metrics["precision"].append(val_precision)
        metrics["recall"].append(val_recall)
        models.append(model)

        if log_mlflow and mlflow.active_run():
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_params({
                    "fold": fold,
                    "train_count": len(train_idx),
                    "val_count": len(val_idx),
                    "best_iteration": model.best_iteration
                })
                mlflow.log_metrics({
                    "val_roc_auc": val_roc_auc,
                    "val_pr_auc": val_pr_auc,
                    "val_precision_0.3": val_precision,
                    "val_recall_0.3": val_recall
                })
                # Log the booster model artifact for this fold
                mlflow.lightgbm.log_model(model, artifact_path="model")

        print(f"Fold {fold} Best Iteration: {model.best_iteration} | Val ROC-AUC: {val_roc_auc:.5f} | Val PR-AUC: {val_pr_auc:.5f} | Val Precision: {val_precision:.5f} | Val Recall: {val_recall:.5f}\n")

        # Clean up before allocating the next fold
        del dtrain, dval, X_val, y_val
        gc.collect()

    return models, oof_preds, metrics
