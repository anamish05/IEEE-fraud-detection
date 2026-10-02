import copy
import torch
import mlflow
import mlflow.pytorch
import dagshub
from src.config import TARGET, TIME_COL, UID_COL, SEED, DAGSHUB_USERNAME,DAGSHUB_REPO 
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def train_and_validate_multimodal(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    save_dir:str,
    run_name: str ,
    scheduler=None,
    epochs: int = 15,
    patience: int = 3,
    params_dict: dict = None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device_type = device.type
    use_amp = (device_type == 'cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    best_ckpt_path = os.path.join(save_dir, f"{run_name}_best.pt")
    oof_pred_path = os.path.join(save_dir, f"{run_name}_oof_preds.npy")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        # Log experiment configuration to dagshub
        if params_dict:
            mlflow.log_params(params_dict)
        mlflow.log_params({
            "epochs": epochs,
            "patience": patience,
            "amp_dtype": str(amp_dtype),
            "device": str(device)
        })
    
        best_val_auc = -1.0
        best_weights = None
        best_epoch=0
        epochs_no_improve = 0

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }

        for epoch in range(1, epochs + 1):
            # -----------------------------------------------------------
            # 1. Training Phase
            # -----------------------------------------------------------
            model.train()
            running_train_loss = 0.0
            total_train_samples = 0

            for x_cont, x_cat, x_seq, seq_mask, x_text, y_batch in train_loader:
                x_cont = x_cont.to(device, non_blocking=True)
                x_cat = x_cat.to(device, non_blocking=True)
                x_seq = x_seq.to(device, non_blocking=True)
                seq_mask = seq_mask.to(device, non_blocking=True)
                x_text = x_text.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Mixed-precision forward pass
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x_cont, x_cat, x_seq, seq_mask, x_text)
                    loss = criterion(logits, y_batch)

                # Scaled backward pass + gradient clipping
                scaler.scale(loss).backward() # multiplies loss by a large factor so gradients don't underflow
                scaler.unscale_(optimizer)  # Prepares real gradient norms for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # scale down to prevent exploding gradients
                scaler.step(optimizer)  # Updates weights only if gradients are valid
                scaler.update()  # adapts S factor up or down for next batch

                running_train_loss += loss.item() * len(y_batch)
                total_train_samples += len(y_batch)

            train_loss = running_train_loss / total_train_samples
            history['train_loss'].append(train_loss)

            # -----------------------------------------------------------
            # 2. Validation Phase
            # -----------------------------------------------------------
            model.eval()
            running_val_loss = 0.0
            total_val_samples = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for x_cont, x_cat, x_seq, seq_mask, x_text, y_batch in val_loader:
                    x_cont = x_cont.to(device, non_blocking=True)
                    x_cat = x_cat.to(device, non_blocking=True)
                    x_seq = x_seq.to(device, non_blocking=True)
                    seq_mask = seq_mask.to(device, non_blocking=True)
                    x_text = x_text.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        logits = model(x_cont, x_cat, x_seq, seq_mask, x_text)
                        loss = criterion(logits, y_batch)

                    # Probabilities via Sigmoid for ROC-AUC
                    probs = torch.sigmoid(logits)

                    running_val_loss += loss.item() * len(y_batch)
                    total_val_samples += len(y_batch)

                    all_val_preds.append(probs.detach().float().cpu().numpy())
                    all_val_targets.append(y_batch.detach().float().cpu().numpy())

            val_loss = running_val_loss / total_val_samples
            y_val_true = np.concatenate(all_val_targets)
            y_val_prob = np.concatenate(all_val_preds)

            if np.isnan(y_val_prob).any():
                print(f"Warning: Found {np.isnan(y_val_prob).sum()} NaNs in validation predictions! Imputing with median.")
                y_val_prob = np.nan_to_num(y_val_prob, nan=0.0)

            val_auc = roc_auc_score(y_val_true, y_val_prob)

            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            # Step LR scheduler if present
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            # Log metrics to MLflow per epoch
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": current_lr
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val ROC-AUC: {val_auc:.5f}"
            )

            # -----------------------------------------------------------
            # 3. Model Checkpointing & Early Stopping
            # -----------------------------------------------------------
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch=epoch
                # Save to Google Drive immediately
                torch.save(best_weights, best_ckpt_path)
                np.save(oof_pred_path, y_val_prob)
                print(f"  --> Checkpoint saved to Google Drive (AUC: {best_val_auc:.5f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Early Stopping] No improvement for {patience} epochs.")
                    break

        # Log final summary metrics and artifacts
            mlflow.log_metric("best_val_auc", best_val_auc)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
            mlflow.log_artifact(oof_pred_path, artifact_path="predictions")
        
        # Load best checkpoint into the returned model
        if best_weights is not None:
            model.load_state_dict(best_weights)

        mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                serialization_format='pickle'
            )
        print(f"Training complete. Dagshub MLflow run logged: {run.info.run_id}")

    return model, history, y_val_prob