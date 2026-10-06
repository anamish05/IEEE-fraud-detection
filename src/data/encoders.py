import json
from pathlib import Path
import pandas as pd


def export_training_artifacts(
    train_df: pd.DataFrame,
    target_col: str,
    te_features: list,
    freq_cols: list,
    features: list,
    artifacts_dir: Path = Path("artifacts"),
    smoothing: float = 10.0,
):
    """Calculates and saves frozen target and frequency encodings from training data."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target encodings
    global_prior = float(train_df[target_col].mean())
    te_maps = {}

    for col in te_features:
        if col in train_df.columns:
            stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"] + smoothing * global_prior
            ) / (stats["count"] + smoothing)
            te_maps[col] = {str(k): float(v) for k, v in smoothed.to_dict().items()}

    with open(artifacts_dir / "target_encodings.json", "w") as f:
        json.dump({"global_prior": global_prior, "mappings": te_maps}, f, indent=2)

    # 2. Frequency encodings
    freq_maps = {}
    for col in freq_cols:
        if col in train_df.columns:
            counts = train_df[col].value_counts(dropna=True).to_dict()
            freq_maps[col] = {str(k): float(v) for k, v in counts.items()}

    with open(artifacts_dir / "freq_encodings.json", "w") as f:
        json.dump(freq_maps, f, indent=2)

    with open(artifacts_dir / "model_features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"Artifacts successfully exported to: {artifacts_dir.resolve()}")