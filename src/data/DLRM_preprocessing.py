import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from typing import Tuple, List, Dict, Optional
import joblib

class DLRMPreprocessor: 
    """
    Stateful preprocessor for DLRM on IEEE-CIS Fraud Detection.
    Fits medians, scalers, and categorical vocabularies strictly on the training set
    and transforms train, validation, or 6-month test splits identically.
    """
    def __init__(
        self,
        numeric_cols: List[str],
        freq_cols: List[str],
        cat_cols: List[str],
        bin_cols: List[str],
        block_bin_flags: List[str],
        min_freq: int = 50,
        scaler_type: str = 'robust',
        clip_val: Optional[float] = 5.0
    ):
        self.numeric_cols = list(numeric_cols)
        self.freq_cols = list(freq_cols)
        self.cat_cols = list(cat_cols)
        self.bin_cols = list(bin_cols)
        self.block_bin_flags = list(block_bin_flags)
        self.min_freq = min_freq
        self.scaler_type = scaler_type
        self.clip_val = clip_val

        self.v_cols: List[str] = []
        self.d_cols: List[str] = []
        self.train_medians: Optional[pd.Series] = None
        self.scaler = None
        self.cols_to_scale: List[str] = []
        self.binary_feature_names: List[str] = []
        self.cont_features: List[str] = []
        self.cat_to_idx: Dict[str, Dict[str, int]] = {}
        self.cardinalities: List[int] = []
        self.has_uid_dt_diff: bool = False
        self._is_fitted: bool = False

        def _engineer_base_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
            X = df.copy()

            # 1. Custom Feature: uid_dt_diff (First txn flag + log transform)
            if is_train:
                self.has_uid_dt_diff = 'uid_dt_diff' in X.columns

            if self.has_uid_dt_diff and 'uid_dt_diff' in X.columns:
                X['uid_dt_diff_is_first'] = X['uid_dt_diff'].isna().astype(np.float32)
                if pd.api.types.is_timedelta64_dtype(X['uid_dt_diff']):
                    X['uid_dt_diff'] = X['uid_dt_diff'].dt.total_seconds()
                diff_clean = X['uid_dt_diff'].fillna(0.0).astype(np.float64).clip(lower=0.0)
                X['uid_dt_diff_log'] = np.log1p(diff_clean).astype(np.float32)

            # 2. Frequency Encodings: log1p Compression
            for col in self.freq_cols:
                if col in X.columns:
                    X[col] = np.log1p(X[col].fillna(0.0).astype(np.float64).clip(lower=0.0)).astype(np.float32)

            # 3. Tri-State & Binary Features (0.0 / 1.0 Flags)
            positive_values = {'T', 'True', True, 'Found', 1, '1.0', 1.0, 'mobile', 'found'}
            if is_train:
                self.binary_feature_names = []

            # Fix Pandas CategoricalDtype deprecation warning on id_15
            if 'id_15' in X.columns:
                X['id_15'] = X['id_15'].astype(object).replace('Unknown', np.nan)

            for col in self.bin_cols:
                if col in X.columns:
                    pos_col = f'{col}_positive'
                    nan_col = f'{col}_nan'
                    X[pos_col] = X[col].isin(positive_values).astype(np.float32)
                    X[nan_col] = X[col].isna().astype(np.float32)
                    if is_train:
                        self.binary_feature_names.extend([pos_col, nan_col])

            for col in self.block_bin_flags:
                if col in X.columns:
                    X[col] = (X[col] == True).astype(np.float32)
                    if is_train:
                        self.binary_feature_names.append(col)

            if self.has_uid_dt_diff and is_train:
                self.binary_feature_names.append('uid_dt_diff_is_first')

            # 4. Continuous Missing Indicators (computed BEFORE median imputation)
            if is_train:
                self.v_cols = [c for c in self.numeric_cols if c.startswith('V')]
                self.d_cols = [c for c in self.numeric_cols if c.startswith('D')]

            X['total_null_count'] = X[self.numeric_cols].isna().sum(axis=1).astype(np.float32)
            X['v_null_count'] = X[self.v_cols].isna().sum(axis=1).astype(np.float32)
            X['d_null_count'] = X[self.d_cols].isna().sum(axis=1).astype(np.float32)

            return X

        def fit(self, X_train_raw: pd.DataFrame) -> "DLRMPreprocessor":
            X_tr = self._engineer_base_features(X_train_raw, is_train=True)
            null_feature_names = ['total_null_count', 'v_null_count', 'd_null_count']

            # Fit medians strictly on training fold
            self.train_medians = X_tr[self.numeric_cols].median().fillna(0.0)
            X_tr[self.numeric_cols] = X_tr[self.numeric_cols].fillna(self.train_medians)

            # Fit continuous scaler
            self.cols_to_scale = list(self.numeric_cols) + list(self.freq_cols) + null_feature_names
            if self.has_uid_dt_diff:
                self.cols_to_scale.append('uid_dt_diff_log')

            if self.scaler_type == 'quantile':
                self.scaler = QuantileTransformer(
                    n_quantiles=1000,
                    output_distribution='normal',
                    random_state=42,
                    subsample=min(200_000, len(X_tr))
                )
            else:
                self.scaler = RobustScaler(with_centering=True, with_scaling=True)

            self.scaler.fit(X_tr[self.cols_to_scale])
            self.cont_features = self.cols_to_scale + self.binary_feature_names

            # Fit Categorical Vocabularies (0 = Missing, 1 = Rare)
            self.cat_to_idx = {}
            self.cardinalities = []

            for col in self.cat_cols:
                tr_series = X_tr[col].astype(str).replace({'nan': 'Missing', 'None': 'Missing', '<NA>': 'Missing'})
                top_cats = tr_series.value_counts()
                frequent_categories = sorted([
                    c for c, count in top_cats.items() 
                    if count >= self.min_freq and c != "Missing"
                ])

                vocab = ["Missing", "Rare"] + frequent_categories
                self.cat_to_idx[col] = {c: i for i, c in enumerate(vocab)}
                self.cardinalities.append(len(vocab))

            self._is_fitted = True
            return self

        def transform(self, X_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            if not self._is_fitted:
                raise RuntimeError("DLRMPreprocessor must be fitted via .fit(X_train) before calling .transform().")

            X = self._engineer_base_features(X_raw, is_train=False)

            # Impute numeric  columns using fitted training statistics
            X[self.numeric_cols] = X[self.numeric_cols].fillna(self.train_medians)

            # Scale continuous columns
            scaled_vals = self.scaler.transform(X[self.cols_to_scale]).astype(np.float32)
            scaled_vals = np.nan_to_num(scaled_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # Clip heavy-tailed outliers so dot-product interactions don't explode
            if self.clip_val is not None:
                scaled_vals = np.clip(scaled_vals, -self.clip_val, self.clip_val)

            X[self.cols_to_scale] = scaled_vals

            # Fast vectorized categorical mapping (unseen & rare values become NaN -> filled with 1 = Rare)
            cat_arrays = []
            for col in self.cat_cols:
                s = X[col].astype(str).replace({'nan': 'Missing', 'None': 'Missing', '<NA>': 'Missing'})
                mapped = s.map(self.cat_to_idx[col]).fillna(1).astype(np.int64).to_numpy()
                cat_arrays.append(mapped)

            X_cat = np.column_stack(cat_arrays) if cat_arrays else np.empty((len(X), 0), dtype=np.int64)
            X_cont = X[self.cont_features].to_numpy(dtype=np.float32)

            return X_cont, X_cat

        def fit_transform(self, X_train_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            return self.fit(X_train_raw).transform(X_train_raw)

        def save(self, filepath: str):
            joblib.dump(self, filepath)

        @staticmethod
        def load(filepath: str) -> "DLRMPreprocessor":
            return joblib.load(filepath)