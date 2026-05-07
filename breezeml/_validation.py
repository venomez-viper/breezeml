import pandas as pd

def check_df_target(df, target):
    """Validate that df is a pandas DataFrame and target is a valid column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
