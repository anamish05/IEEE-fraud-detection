import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from typing import Tuple, List


def prepare_nn_fold_data(
    X_train_raw: pd.DataFrame, 
    X_val_raw: pd.DataFrame, 
    numeric_cols: list, 
    freq_cols: list,
    cat_cols: list, 
    bin_cols: list,
    block_bin_flags: list,
    te_cols: list,
    min_freq: int = 50,
    scaler_type: str = 'robust'
):
    """
    Fits imputation, log transforms, scaling, and categorical vocabularies
    strictly on the training fold, transforming both train and validation splits.
    
    Returns:
        X_tr_cont, X_va_cont : np.ndarray (float32) for Linear / BatchNorm layers
        X_tr_cat,  X_va_cat  : np.ndarray (int64) for nn.Embedding layers
        embedding_sizes      : list of (num_classes, emb_dim)
        cont_features        : list of continuous feature names
    """
    X_tr = X_train_raw.copy()
    X_va = X_val_raw.copy()

    # -------------------------------------------------------------------------
    # 1. Custom Feature: uid_dt_diff (First txn flag + log transform)
    # -------------------------------------------------------------------------
    if 'uid_dt_diff' in X_tr.columns:
        # Binary flag: 1.0 if this is the first transaction (was NaN), 0.0 otherwise
        X_tr['uid_dt_diff_is_first'] = X_tr['uid_dt_diff'].isna().astype(np.float32)
        X_va['uid_dt_diff_is_first'] = X_va['uid_dt_diff'].isna().astype(np.float32)

        # Convert Timedelta to total seconds if not already numeric
        for df_slice in [X_tr, X_va]:
            if pd.api.types.is_timedelta64_dtype(df_slice['uid_dt_diff']):
                df_slice['uid_dt_diff'] = df_slice['uid_dt_diff'].dt.total_seconds()

        # Impute NaN with 0.0, guard against negative offsets, then apply log1p
        tr_diff_clean = X_tr['uid_dt_diff'].fillna(0.0).astype(np.float64).clip(lower=0.0)
        va_diff_clean = X_va['uid_dt_diff'].fillna(0.0).astype(np.float64).clip(lower=0.0)

        X_tr['uid_dt_diff_log'] = np.log1p(tr_diff_clean).astype(np.float32)
        X_va['uid_dt_diff_log'] = np.log1p(va_diff_clean).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. Frequency Encodings: log1p Compression
    # -------------------------------------------------------------------------
    for col in freq_cols:
        if col in X_tr.columns:
            X_tr[col] = np.log1p(X_tr[col].fillna(0.0).clip(lower=0.0)).astype(np.float32)
            X_va[col] = np.log1p(X_va[col].fillna(0.0).clip(lower=0.0)).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Tri-State & Binary Features (0.0 / 1.0 Flags)
    # -------------------------------------------------------------------------
    positive_values = {'T', 'True', True, 'Found', 1, '1.0', 1.0, 'mobile', 'found'}
    binary_feature_names = []

    # Clean known anomalous tokens
    if 'id_15' in X_tr.columns:
        X_tr['id_15'] = X_tr['id_15'].replace('Unknown', np.nan)
        X_va['id_15'] = X_va['id_15'].replace('Unknown', np.nan)

    for col in bin_cols:
        if col in X_tr.columns:
            pos_col = f'{col}_positive'
            nan_col = f'{col}_nan'

            X_tr[pos_col] = X_tr[col].isin(positive_values).astype(np.float32)
            X_tr[nan_col] = X_tr[col].isna().astype(np.float32)

            X_va[pos_col] = X_va[col].isin(positive_values).astype(np.float32)
            X_va[nan_col] = X_va[col].isna().astype(np.float32)

            binary_feature_names.extend([pos_col, nan_col])

    # Convert existing boolean flags to clean float32
    for col in block_bin_flags:
        if col in X_tr.columns:
            X_tr[col] = (X_tr[col] == True).astype(np.float32)
            X_va[col] = (X_va[col] == True).astype(np.float32)
            binary_feature_names.append(col)

    # Add the newly created first-transaction flag to unscaled binary list
    if 'uid_dt_diff_is_first' in X_tr.columns:
        binary_feature_names.append('uid_dt_diff_is_first')

    # -------------------------------------------------------------------------
    # 4. Continuous Missing Indicators & Imputation
    # -------------------------------------------------------------------------
    v_cols = [c for c in numeric_cols if c.startswith('V')]
    d_cols = [c for c in numeric_cols if c.startswith('D')]

    X_tr['total_null_count'] = X_tr[numeric_cols].isna().sum(axis=1).astype(np.float32)
    X_tr['v_null_count'] = X_tr[v_cols].isna().sum(axis=1).astype(np.float32)
    X_tr['d_null_count'] = X_tr[d_cols].isna().sum(axis=1).astype(np.float32)

    X_va['total_null_count'] = X_va[numeric_cols].isna().sum(axis=1).astype(np.float32)
    X_va['v_null_count'] = X_va[v_cols].isna().sum(axis=1).astype(np.float32)
    X_va['d_null_count'] = X_va[d_cols].isna().sum(axis=1).astype(np.float32)

    null_feature_names = ['total_null_count', 'v_null_count', 'd_null_count']

    # Impute numeric NaNs strictly with training medians
    train_medians = X_tr[numeric_cols].median().fillna(0.0)
    X_tr[numeric_cols] = X_tr[numeric_cols].fillna(train_medians)
    X_va[numeric_cols] = X_va[numeric_cols].fillna(train_medians)

    # Impute Target Encoding NaNs with training split mean
    if len(te_cols) > 0:
        te_fill = X_tr[te_cols].mean().fillna(0.035)
        X_tr[te_cols] = X_tr[te_cols].fillna(te_fill)
        X_va[te_cols] = X_va[te_cols].fillna(te_fill)

    # -------------------------------------------------------------------------
    # 5. Continuous Feature Scaling
    # -------------------------------------------------------------------------
    # All continuous features requiring distribution scaling
    cols_to_scale = list(numeric_cols) + list(freq_cols) + null_feature_names
    if 'uid_dt_diff_log' in X_tr.columns:
        cols_to_scale.append('uid_dt_diff_log')

    if scaler_type == 'quantile':
        scaler = QuantileTransformer(
            n_quantiles=1000, 
            output_distribution='normal', 
            random_state=42, 
            subsample=min(200_000, len(X_tr))
        )
    else:
        scaler = RobustScaler(with_centering=True, with_scaling=True)

    X_tr[cols_to_scale] = scaler.fit_transform(X_tr[cols_to_scale]).astype(np.float32)
    X_va[cols_to_scale] = scaler.transform(X_va[cols_to_scale]).astype(np.float32)

    # Sanitize any division-by-zero infinities
    X_tr[cols_to_scale] = X_tr[cols_to_scale].replace([np.inf, -np.inf], 0.0)
    X_va[cols_to_scale] = X_va[cols_to_scale].replace([np.inf, -np.inf], 0.0)

    # -------------------------------------------------------------------------
    # 6. Categorical Vocabulary Mapping for nn.Embedding
    # -------------------------------------------------------------------------
    embedding_sizes = []

    for col in cat_cols:
        tr_series = X_tr[col].astype('object').fillna("Missing").astype(str)
        va_series = X_va[col].astype('object').fillna("Missing").astype(str)

        # Identify frequent tokens strictly on the train slice
        top_cats = tr_series.value_counts()
        frequent_categories = set(top_cats[top_cats >= min_freq].index)
        frequent_categories.discard("Missing")

        # Map rare strings to 'Rare'
        tr_clean = tr_series.apply(lambda x: x if (x in frequent_categories or x == "Missing") else "Rare")
        va_clean = va_series.apply(lambda x: x if (x in frequent_categories or x == "Missing") else "Rare")

        # Build stable indexing dictionary
        unique_categories = ["Missing", "Rare"] + sorted(list(frequent_categories))
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}

        # Integer encode; unseen categories default to index 1 ('Rare')
        X_tr[col] = tr_clean.map(cat_to_idx).fillna(1).astype(np.int64)
        X_va[col] = va_clean.map(cat_to_idx).fillna(1).astype(np.int64)

        # Dimension rule: min(60, ceil(sqrt(cardinality)))
        num_classes = len(unique_categories)
        emb_dim = max(2, min(60, int(np.ceil(np.sqrt(num_classes)))))
        embedding_sizes.append((num_classes, emb_dim))

    # -------------------------------------------------------------------------
    # 7. Final Output Extraction
    # -------------------------------------------------------------------------
    cont_features = cols_to_scale + te_cols + binary_feature_names

    X_tr_cont = X_tr[cont_features].values.astype(np.float32)
    X_va_cont = X_va[cont_features].values.astype(np.float32)

    X_tr_cat = X_tr[cat_cols].values.astype(np.int64)
    X_va_cat = X_va[cat_cols].values.astype(np.int64)

    return X_tr_cont, X_va_cont, X_tr_cat, X_va_cat, embedding_sizes, cont_features





def build_vectorized_user_sequences(
    df: pd.DataFrame,
    uid_col:str,
    time_col: str,
    seq_feature_cols: List[str],
    max_len: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs past transaction history tensors for each transaction in df
    without Python row-loops.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Source DataFrame containing all transactions.
    uid_col : str
        Column name for user/card proxy ID.
    time_col : str
        Column name used for temporal ordering.
    seq_feature_cols : List[str]
        List of continuous feature names to include in each sequence time step.
    max_len : int
        Maximum number of transactions to look back (including current). 
        Default 15 balances burst detection and Transformer compute.
        
    Returns:
    --------
    X_seq : np.ndarray, shape (N, max_len, D_seq), dtype float32
        3D tensor containing sequence features per transaction.
    padding_mask : np.ndarray, shape (N, max_len), dtype bool
        Boolean mask where True indicates padded (non-existent past) transactions,
        conforming directly to PyTorch's `src_key_padding_mask` convention.
    """
    if seq_feature_cols is None:
        raise ValueError("Must provide at least one feature column for seq_feature_cols.")
        
    n_samples = len(df)
    n_feats = len(seq_feature_cols)

    # 1. Preserve original row index so output matches the input DataFrame order
    df_work = df[[uid_col, time_col] + seq_feature_cols].copy()
    df_work['__orig_idx__'] = np.arange(n_samples)

    # 2. Sort by User and Time
    # Note: NaN UIDs are isolated and will not link sequences across unrelated transactions
    df_work = df_work.sort_values(by=[uid_col, time_col], kind='mergesort').reset_index(drop=True)

    # 3. Calculate sequence position within each user group
    # Transactions with NaN UID get isolated to individual length-1 sequences
    is_valid_uid = df_work[uid_col].notna().values # false if NaN
    
    # Fast cumulative count per UID
    uid_codes, _ = pd.factorize(df_work[uid_col].fillna("__NULL_UID__")) # get integer codes
    same_as_prev = (uid_codes[1:] == uid_codes[:-1]) & is_valid_uid[1:] & is_valid_uid[:-1]
    
    # Step-based cumcount using boundary detection
    boundary_marker = np.concatenate([[True], ~same_as_prev])  # a new group begins where same_as_prev is False
    group_starts = np.flatnonzero(boundary_marker) # indexes of non zero elements
    group_lengths = np.diff(np.concatenate([group_starts, [n_samples]])) # distance between successive start indices
    cum_counts = np.arange(n_samples) - np.repeat(group_starts, group_lengths) # resets the counter back to 0 at each boundary

    # 4. Build Backward-Looking Matrix of Indices (Shape: N x max_len)
    # lag 0 is current step, lag 1 is t-1, ..., lag (max_len - 1) is t-(max_len-1)
    current_sorted_indices = np.arange(n_samples)
    lags = np.arange(max_len - 1, -1, -1)  # Left-to-right chronological order: [t-(L-1), ..., t-1, t]

    # Shifted indices: current_index - lag
    candidate_indices = current_sorted_indices[:, None] - lags[None, :]

    # Determine valid past steps within the same user boundary:
    # A step is valid if: lag <= cum_counts (meaning we don't bleed into prior user)
    valid_mask = lags[None, :] <= cum_counts[:, None]

    # Map invalid/padded slots to index 0 (safe dummy row for array indexing)
    safe_indices = np.where(valid_mask, candidate_indices, 0)

    # 5. Extract Feature Matrix with a Dummy Zero Row
    # Raw feature matrix from sorted data
    raw_feats = df_work[seq_feature_cols].to_numpy(dtype=np.float32)
    # Replace any accidental NaN with 0.0 before tensor construction
    raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # Append dummy row at index 0 for all masked positions
    dummy_feats = np.vstack([np.zeros((1, n_feats), dtype=np.float32), raw_feats])
    # Offset safe_indices by +1 because row 0 is now the dummy zero row
    shifted_indices = np.where(valid_mask, safe_indices + 1, 0)

    # 6. Gather into 3D Tensor
    # Shape: (N, max_len, D_seq)
    X_seq_sorted = dummy_feats[shifted_indices]

    # 7. Invert valid_mask to PyTorch padding_mask (True = PAD / IGNORE)
    # Shape: (N, max_len)
    padding_mask_sorted = ~valid_mask

    # 8. Re-align arrays back to the original DataFrame row order
    orig_order = df_work['__orig_idx__'].to_numpy()
    
    X_seq = np.empty_like(X_seq_sorted)
    padding_mask = np.empty_like(padding_mask_sorted)

    X_seq[orig_order] = X_seq_sorted
    padding_mask[orig_order] = padding_mask_sorted

    return X_seq, padding_mask