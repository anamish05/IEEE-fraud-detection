
import numpy as np
import pandas as pd
from src.config import  TARGET, TIME_COL, UID_COL, SEED
from sklearn.model_selection import KFold


def rolling_24h_stats_per_user(df):
    """This script calculates rolling 24-hour historical transaction statistics (count and average amount) per user without 
    data leakage (it looks strictly at past transactions, excluding the current one).
    Instead of using slow pandas operations like df.groupby(...).rolling('24h'), it converts the data into raw 1D NumPy arrays 
    and uses binary search (np.searchsorted) and prefix sums (np.cumsum) to compute these rolling metrics 
    at $O(N log K)$ speed with minimal memory overhead."""

    # Temporal Velocity Aggregations (Before Split)

    # Sort dataframe by UID and TransactionDT (in-place to save memory)
    df.sort_values([UID_COL, 'datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract minimal 1D arrays
    uids = df[UID_COL].values
    times = df['datetime'].values
    amts = df['TransactionAmt'].values

    n = len(df)
    counts = np.zeros(n, dtype=np.int32)
    means = np.full(n, np.nan, dtype=np.float32)

    # Find slice boundaries where Pseudo_UID changes
    change_idx = np.flatnonzero(uids[:-1] != uids[1:]) + 1
    group_starts = np.concatenate(([0], change_idx))
    group_ends = np.concatenate((change_idx, [n]))

    WINDOW_SEC = 24 * 3600  # 86,400 seconds

    # Fast C-level searchsorted per user
    for start, end in zip(group_starts, group_ends):
        t_group = times[start:end]
        a_group = amts[start:end]
        group_len = end - start

        # Prefix sums for O(1) interval sum
        prefix_sums = np.concatenate(([0.0], np.cumsum(a_group)))

        # Find the earliest index in window for each transaction: [t - 86400, t)
        # searchsorted('left') on (t_group - WINDOW_SEC) gives the start index
        left_indices = np.searchsorted(t_group, t_group - WINDOW_SEC, side='left')
        current_indices = np.arange(group_len)

        # Number of prior transactions in [t - 24h, t)
        cnt = current_indices - left_indices
        counts[start:end] = cnt

        # Window sum = prefix[current] - prefix[left]
        window_sums = prefix_sums[current_indices] - prefix_sums[left_indices]

        # Avoid div by 0; where count > 0, compute mean
        has_history = cnt > 0
        means_slice = np.full(group_len, np.nan, dtype=np.float32)
        means_slice[has_history] = window_sums[has_history] / cnt[has_history]
        means[start:end] = means_slice

    # 5. Direct 1D assignment (zero index copying)
    df['uid_count_24h'] = counts     # count of transactions by the user in prior 24 h
    df['uid_amt_mean_24h'] = np.where(np.isnan(means), amts, means)  # average transaction amount spent by the user in prior 24 h


    # Clean up raw arrays
    del uids, times, amts, counts, means, group_starts, group_ends

    return df



#  Leakage-Free Out-Of-Fold Target Encoding Function
# Execute Target Encoding on selected high-cardinality features


def apply_oof_target_encoding_by_index(
    df: pd.DataFrame, 
    train_idx: np.ndarray, 
    val_idx: np.ndarray, 
    target_col: str, 
    encode_cols: list, 
    n_splits: int = 5, 
    smoothing: int = 10
):
    """
    Computes smoothed target encoding in-place on `df` using index arrays.
    Prevents RAM duplication of the full DataFrame.
    """
    # 1. Global prior strictly computed on training rows only
    y_train = df.iloc[train_idx][target_col]
    global_mean = float(y_train.mean())
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    for c in encode_cols:
        enc_col_name = f'{c}_target_enc'
        
        # Initialize an empty float32 array for the whole column
        enc_values = np.full(len(df), np.nan, dtype=np.float32)
        
        # -------------------------------------------------------------
        # Step A: OOF Encoding for Train Split (5-fold inside train_idx)
        # -------------------------------------------------------------
    
        for fold_tr_pos, fold_oof_pos in kf.split(train_idx):
            tr_indices = train_idx[fold_tr_pos]
            oof_indices = train_idx[fold_oof_pos]
            
            # Group statistics using only this fold's training portion
            fold_tr_slice = df.iloc[tr_indices][[c, target_col]]
            stats = fold_tr_slice.groupby(c, observed=False)[target_col].agg(['count', 'mean'])
            
            smoothed_fold = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            # Map values to the out-of-fold slice
            oof_cats = df.iloc[oof_indices][c]
            enc_values[oof_indices] = oof_cats.map(smoothed_fold).fillna(global_mean).values
            
        # -------------------------------------------------------------
        # Step B: Validation Mapping (using 100% of train_idx stats)
        # -------------------------------------------------------------
        full_tr_slice = df.iloc[train_idx][[c, target_col]]
        stats_full = full_tr_slice.groupby(c, observed=False)[target_col].agg(['count', 'mean'])
        smoothed_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
        
        val_cats = df.iloc[val_idx][c]
        enc_values[val_idx] = val_cats.map(smoothed_full).fillna(global_mean).values
        
        # Assign directly into the dataframe
        df[enc_col_name] = enc_values
        
    return df
