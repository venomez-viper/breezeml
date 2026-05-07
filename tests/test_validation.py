import pytest
import pandas as pd
from breezeml._validation import check_df_target
from breezeml import fit

def test_check_df_target_valid():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    # Should not raise an exception
    check_df_target(df, "a")

def test_check_df_target_invalid_type():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        check_df_target([1, 2, 3], "a")

def test_check_df_target_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Target column 'b' not found"):
        check_df_target(df, "b")

def test_fit_validation():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Target column 'b' not found"):
        fit(df, "b")
