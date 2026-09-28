import numpy as np
import pandas as pd

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Safely downcasts numeric columns without crashing on strings or datetimes."""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        # Strictly skip non-numeric types (strings, categories, datetimes, booleans)
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # If the column is entirely NaN or empty, skip downcasting
        if pd.isna(c_min) or pd.isna(c_max):
            continue

        # Integer downcasting
        if pd.api.types.is_integer_dtype(df[col]):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Float downcasting (stay on float32 to avoid float16 precision/LightGBM bugs)
        elif pd.api.types.is_float_dtype(df[col]):
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage decreased to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df