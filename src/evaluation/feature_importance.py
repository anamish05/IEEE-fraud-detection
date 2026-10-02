
from pathlib import Path
from typing import List, Optional
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def feature_importance(
        models: List[lgb.Booster],
        features: List[str],
        importance_type: str='gain'
) -> pd.DataFrame:

    records=[]

    for fold_idx, model in enumerate(models, 1):
        # 'gain' measures total split gain contribution across trees
        gains=model.feature_importance(importance_type=importance_type)
        for feature, val in zip(features, gains):
            records.append(
                {'feature': feature, 'importance': val, 'fold': fold_idx}
            )
       
    fi_df = pd.DataFrame(records)

    # Compute Mean and Standard Deviation per Feature

    agg_importance = (
        fi_df.groupby('feature')['importance']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'mean_gain', 'std': 'std_gain'})
    )

    # Stability metric: Coefficient of Variation (lower = more temporally stable)
    agg_importance['std_gain'] = agg_importance['std_gain'].fillna(0)
    agg_importance['stability_ratio'] = agg_importance['std_gain'] / (agg_importance['mean_gain'] + 1e-6)

    # Sort by top contributors
    agg_importance = agg_importance.sort_values(by='mean_gain', ascending=False).reset_index(drop=True)

    return agg_importance