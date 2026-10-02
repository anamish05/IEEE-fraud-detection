import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

# rank normalization of the prediction probabilities 
# blend together predictions 

def rank_norm_blend_models(
        lgb_val_preds: np.ndarray, 
        nn_val_preds: np.ndarray, 
        y_val: np.ndarray):
    
    rank_lgb = (rankdata(lgb_val_preds) - 1.0) / (len(lgb_val_preds) - 1.0)
    rank_nn = (rankdata(nn_val_preds) - 1.0) / (len(nn_val_preds) - 1.0)

    # 2. Compute prediction diversity / correlation
    pred_corr = np.corrcoef(rank_lgb, rank_nn)[0, 1]
    print(f"Rank Correlation between LGBM and NN: {pred_corr:.4f}")
    print("Note: Lower correlation (< 0.90) indicates high diversity and maximum ensemble lift.")

    # 3. Standalone baselines
    lgb_auc = roc_auc_score(y_val, rank_lgb)
    lgb_prauc = average_precision_score(y_val, rank_lgb)

    nn_auc = roc_auc_score(y_val, rank_nn)
    nn_prauc = average_precision_score(y_val, rank_nn)

    print("\n" + "=" * 55)
    print(f"LGBM Standalone | ROC-AUC: {lgb_auc:.5f} | PR-AUC: {lgb_prauc:.5f}")
    print(f"NN   Standalone | ROC-AUC: {nn_auc:.5f} | PR-AUC: {nn_prauc:.5f}")
    print("=" * 55)

    # 4. Grid Search Weights
    best_auc = -1.0
    best_weight = None
    results = []

    for w_lgb in np.linspace(0.0, 1.0, 21):
        w_nn = 1.0 - w_lgb
        blend = (w_lgb * rank_lgb) + (w_nn * rank_nn)
        
        auc = roc_auc_score(y_val, blend)
        prauc = average_precision_score(y_val, blend)
        delta = auc - lgb_auc
        
        results.append({
            'w_lgb': w_lgb,
            'w_nn': w_nn,
            'auc': auc,
            'prauc': prauc,
            'delta_vs_lgb': delta
        })
        
        if auc > best_auc:
            best_auc = auc
            best_weight = (w_lgb, w_nn)

    results_df = pd.DataFrame(results)
    delta_vs_lgb = best_auc - lgb_auc

    return results_df, best_weight, best_auc, delta_vs_lgb