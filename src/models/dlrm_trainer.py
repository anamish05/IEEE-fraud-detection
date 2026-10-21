import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
import mlflow.pytorch
from typing import Dict, List, Tuple

from src.models.dlrm import DLRM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_dlrm(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for x_cont, x_cat, y in val_loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x_cont, x_cat)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(y.cpu().numpy())

    val_loss = total_loss / n_samples
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    val_roc_auc = roc_auc_score(targets, preds)
    val_pr_auc = average_precision_score(targets, preds)

    return val_loss, val_pr_auc, val_roc_auc, preds


def train_single_dlrm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    patience: int,
    device: torch.device,
    save_path: str
) -> Tuple[nn.Module, Dict[str, List[float]], np.ndarray]:
    best_pr_auc = -1.0
    best_roc_auc = -1.0
    best_preds = None
    best_weights = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_roc_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0

        for x_cont, x_cat, y in train_loader:
            x_cont = x_cont.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n_train += bs

        train_loss = running_loss / n_train
        val_loss, val_pr_auc, val_roc_auc, val_preds = evaluate_dlrm(
            model, val_loader, criterion, device
        )

        # Step scheduler on Validation PR-AUC
        scheduler.step(val_pr_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)

        # Log epoch metrics to active MLflow run
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc
            }, step=epoch)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.5f} | Val ROC-AUC: {val_roc_auc:.5f}"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_roc_auc = val_roc_auc
            best_preds = val_preds
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_weights, save_path)
            print(f"  --> Best checkpoint saved (PR-AUC: {best_pr_auc:.5f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] No PR-AUC improvement for {patience} epochs.")
                break

    model.load_state_dict(best_weights)
    return model, history, best_preds

# train DLRM on different seeds to lower variance and then average them to return more stable and reliable result, not to depend on random initialization
def train_dlrm_seed_ensemble(
    X_tr_cont: np.ndarray,
    X_tr_cat: np.ndarray,
    y_train: np.ndarray,
    X_va_cont: np.ndarray,
    X_va_cat: np.ndarray,
    y_val: np.ndarray,
    cardinalities: List[int],
    save_dir: str,
    run_name: str = "dlrm_seed_ensemble",
    seeds: List[int] = [42, 100, 2026],
    d: int = 16,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    patience: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[List[nn.Module], np.ndarray]:
    """
    Trains DLRM across multiple random seeds, logs child runs and ensemble metrics
    to MLflow, and returns the list of trained models + seed-averaged validation predictions.
    """
    from src.models.dlrm import FraudDLRMDataset

    os.makedirs(save_dir, exist_ok=True)
    val_dataset = FraudDLRMDataset(X_va_cont, X_va_cat, y_val)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight_val = float(n_neg / n_pos) ** 0.5
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    models = []
    seed_preds = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "architecture": "DLRM",
            "d_embed": d,
            "n_dense": X_tr_cont.shape[1],
            "n_cat": len(cardinalities),
            "seeds": str(seeds),
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "pos_weight": round(pos_weight_val, 4)
        })

        for seed in seeds:
            print(f"\n{'='*60}\nTraining DLRM | Seed {seed}\n{'='*60}")
            set_seed(seed)

            # Generator ensures deterministic batch shuffling per seed
            g = torch.Generator()
            g.manual_seed(seed)
            train_dataset = FraudDLRMDataset(X_tr_cont, X_tr_cat, y_train)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g
            )

            model = DLRM(
                n_dense=X_tr_cont.shape[1],
                cardinalities=cardinalities,
                d=d
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-5
            )

            ckpt_path = os.path.join(save_dir, f"{run_name}_seed{seed}.pt")

            with mlflow.start_run(run_name=f"{run_name}_seed{seed}", nested=True):
                mlflow.log_param("seed", seed)
                trained_model, hist, best_val_preds = train_single_dlrm(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    patience=patience,
                    device=device,
                    save_path=ckpt_path
                )
                seed_pr = average_precision_score(y_val, best_val_preds)
                seed_roc = roc_auc_score(y_val, best_val_preds)
                mlflow.log_metrics({"best_val_pr_auc": seed_pr, "best_val_roc_auc": seed_roc})

            models.append(trained_model)
            seed_preds.append(best_val_preds)

        # Compute Seed-Averaged Ensemble Metrics
        ensemble_preds = np.mean(seed_preds, axis=0)
        ens_pr_auc = average_precision_score(y_val, ensemble_preds)
        ens_roc_auc = roc_auc_score(y_val, ensemble_preds)

        mlflow.log_metrics({
            "ensemble_val_pr_auc": ens_pr_auc,
            "ensemble_val_roc_auc": ens_roc_auc
        })

        print(f"\n{'='*60}")
        print(f"Seed Ensemble Complete ({len(seeds)} seeds)")
        print(f"Ensemble Val PR-AUC:  {ens_pr_auc:.5f}")
        print(f"Ensemble Val ROC-AUC: {ens_roc_auc:.5f}")
        print(f"{'='*60}")

    return models, ensemble_preds