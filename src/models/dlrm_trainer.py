import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
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
    device: torch.device,
    thresh: float
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
    val_precision = precision_score(targets, (preds >= thresh).astype(int), zero_division=0)
    val_recall = recall_score(targets, (preds >= thresh).astype(int), zero_division=0)

    return val_loss, val_pr_auc, val_roc_auc, val_precision, val_recall, preds


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

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_roc_auc": [], "val_precision": [], "val_recall": []}

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # preserve gradients from exploding
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            n_train += batch_size

        train_loss = running_loss / n_train
        val_loss, val_pr_auc, val_roc_auc, val_precision, val_recall, val_preds = evaluate_dlrm(
            model, val_loader, criterion, device, thresh=0.2
        )

        # Step scheduler on Validation PR-AUC - dynamically lowers lr when model stops making progress on val set
        scheduler.step(val_pr_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        # Log epoch metrics to active MLflow run
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_precision": val_precision,
                "val_recall": val_recall
            }, step=epoch)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.5f} | Val ROC-AUC: {val_roc_auc:.5f}"
            f"Val Precision: {val_precision:.5f} | Val Recall: {val_recall:.5f}"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
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

