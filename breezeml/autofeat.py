"""
Automatic feature engineering: turn a raw DataFrame into a richer, model-ready
one and explain every transformation along the way.

    new_df, report = breezeml.autofeat.engineer(df, "target")

This is the higher-level companion to ``breezeml.features``. Where ``features``
offers individual helpers (select, pca, polynomial), ``engineer`` runs a whole
opinionated pipeline in one call and hands back a "what I did and why" report:

    1. Datetime expansion  - split date columns into year/month/day/... parts.
    2. High-cardinality encoding - frequency + leakage-safe out-of-fold target
       means for categoricals with too many levels to one-hot.
    3. Numeric interactions - a capped set of pairwise products among the most
       informative numeric columns.
    4. Pruning - drop constant columns and one of each near-duplicate numeric
       pair.

Everything is leakage-safe (target encoding uses K-fold out-of-fold means) and
every step is bounded by an explicit cap so the feature count can never explode.
The input DataFrame is never mutated; a brand new one is returned with the
target column preserved untouched.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ._validation import check_df_target

__all__ = ["engineer"]

# --- caps and thresholds (documented so nothing can explode silently) --------
HIGH_CARD_MIN_UNIQUE = 20        # > this many distinct levels counts as high-card
HIGH_CARD_ROW_SHARE = 0.05       # ...or more distinct levels than 5% of the rows
MAX_INTERACTION_BASE = 5         # only the top-N numeric columns feed interactions
MAX_INTERACTION_COLS = 10        # hard cap on new interaction columns created
CORR_DROP_THRESHOLD = 0.98       # |r| above this = near-duplicate, drop one
DATETIME_PARSE_MIN_RATE = 0.9    # object col must parse this cleanly to be a date
N_OOF_FOLDS = 5                  # folds for out-of-fold target encoding


def _numeric_target(y: pd.Series):
    """Return (numeric_target, ok). Target-mean encoding needs a numeric target.

    Numeric and boolean targets pass straight through. A binary non-numeric
    target is mapped to 0/1. Anything else (multiclass strings) cannot be
    target-mean encoded meaningfully, so we report ok=False and skip it.
    """
    if pd.api.types.is_bool_dtype(y):
        return y.astype(float), True
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float), True
    if y.nunique(dropna=True) == 2:
        codes, _ = pd.factorize(y)
        return pd.Series(codes.astype(float), index=y.index), True
    return None, False


def _is_datetimey(series: pd.Series) -> bool:
    """True if the column is already datetime dtype or parses cleanly as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Only attempt to parse object/string columns; never coerce numerics (an
    # integer id column must not be mistaken for epoch timestamps).
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(non_null, errors="coerce")
    return parsed.notna().mean() >= DATETIME_PARSE_MIN_RATE


def _expand_datetime(series: pd.Series, name: str):
    """Return (dict_of_new_columns, list_of_new_names) from a datetime column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = pd.to_datetime(series, errors="coerce")
    new_cols: dict[str, pd.Series] = {}
    new_cols[f"{name}_year"] = s.dt.year
    new_cols[f"{name}_month"] = s.dt.month
    new_cols[f"{name}_day"] = s.dt.day
    new_cols[f"{name}_dayofweek"] = s.dt.dayofweek
    # Only add hour when the column actually carries a time of day; a pure date
    # column would give an all-zero, useless hour feature.
    has_time = bool(((s.dt.hour != 0) | (s.dt.minute != 0) | (s.dt.second != 0)).any())
    if has_time:
        new_cols[f"{name}_hour"] = s.dt.hour
    new_cols[f"{name}_is_weekend"] = (s.dt.dayofweek >= 5).astype("float")
    return new_cols, list(new_cols.keys())


def _oof_target_encoding(cat: pd.Series, y_num: pd.Series, n_splits: int) -> pd.Series:
    """Leakage-safe target-mean encoding via out-of-fold K-fold means.

    A naive target encoding maps each level to the mean target over ALL rows,
    including the row being encoded, which leaks the label straight into the
    feature. Instead we split into K folds and, for each fold, compute the
    level means from the OTHER folds only. A row's own target never touches its
    own encoding. Unseen levels fall back to the global mean.
    """
    n = len(cat)
    n_splits = max(2, min(n_splits, n))
    enc = np.full(n, np.nan, dtype=float)
    cat_arr = cat.astype("object").to_numpy()
    y_arr = y_num.to_numpy(dtype=float)
    global_mean = float(np.nanmean(y_arr))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(cat_arr):
        fold = pd.DataFrame({"c": cat_arr[train_idx], "y": y_arr[train_idx]})
        means = fold.groupby("c")["y"].mean()
        mapped = pd.Series(cat_arr[val_idx]).map(means).to_numpy(dtype=float)
        mapped = np.where(np.isnan(mapped), global_mean, mapped)
        enc[val_idx] = mapped
    return pd.Series(enc, index=cat.index)


def _rank_numeric_columns(work: pd.DataFrame, cols: list, y_num, y_ok: bool) -> list:
    """Order numeric columns by informativeness: |corr with target|, else variance."""
    if not cols:
        return []
    if y_ok:
        scores = {}
        for c in cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = np.corrcoef(work[c].astype(float), y_num.astype(float))[0, 1]
            scores[c] = 0.0 if np.isnan(r) else abs(float(r))
    else:
        scores = {c: float(work[c].var()) if pd.notna(work[c].var()) else 0.0 for c in cols}
    return sorted(cols, key=lambda c: scores[c], reverse=True)


def engineer(
    df: pd.DataFrame,
    target: str,
    datetime_cols: list | None = None,
    show: bool = True,
):
    """Automatically engineer a richer feature set and explain every step.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training data including the target column. Never mutated.
    target : str
        Target column name. Preserved untouched in the returned frame.
    datetime_cols : list of str, optional
        Columns to force-treat as datetimes even if auto-detection misses them.
    show : bool
        Print a human-readable summary of what was done and why (default True).

    Returns
    -------
    (new_df, report) : tuple
        ``new_df`` is a brand new DataFrame with the target preserved. ``report``
        is a dict: ``added``, ``dropped``, ``encoded``, ``datetime_expanded``,
        ``n_features_before``, ``n_features_after``, ``notes``.
    """
    check_df_target(df, target)
    datetime_cols = list(datetime_cols) if datetime_cols else []

    work = df.copy()
    y = work[target]
    y_num, y_ok = _numeric_target(y)

    n_features_before = work.shape[1] - 1  # exclude the target
    added: list = []
    dropped: list = []
    encoded: list = []
    datetime_expanded: list = []
    notes: list = []
    generated_encoding_cols: set = set()

    feature_cols = [c for c in work.columns if c != target]

    # --- Step 1: datetime expansion -----------------------------------------
    dt_targets = []
    for c in feature_cols:
        if c in datetime_cols or _is_datetimey(work[c]):
            dt_targets.append(c)
    for c in dt_targets:
        new_cols, names = _expand_datetime(work[c], c)
        for name, series in new_cols.items():
            work[name] = series.values
        work = work.drop(columns=[c])
        added.extend(names)
        datetime_expanded.append({"column": c, "into": names})
    if datetime_expanded:
        notes.append(
            "Datetime columns were split into calendar parts (year/month/day/"
            "dayofweek/is_weekend, plus hour when a time of day was present) and "
            "the raw datetime column dropped."
        )

    # --- Step 2: high-cardinality categorical encoding ----------------------
    cat_cols = [
        c for c in work.columns
        if c != target
        and not pd.api.types.is_numeric_dtype(work[c])
        and not pd.api.types.is_datetime64_any_dtype(work[c])
        and not pd.api.types.is_bool_dtype(work[c])
    ]
    n_rows = len(work)
    used_oof = False
    for c in cat_cols:
        n_unique = work[c].nunique(dropna=True)
        is_high_card = n_unique > HIGH_CARD_MIN_UNIQUE or n_unique > HIGH_CARD_ROW_SHARE * n_rows
        if not is_high_card:
            continue  # low-cardinality stays as-is for normal one-hot downstream

        encodings = []
        # frequency encoding: each level -> its share of the rows
        shares = work[c].map(work[c].value_counts(normalize=True))
        freq_name = f"{c}_freq"
        work[freq_name] = shares.astype("float").values
        added.append(freq_name)
        generated_encoding_cols.add(freq_name)
        encodings.append("frequency")

        method = "frequency-only (target not numeric/binary, target encoding skipped)"
        if y_ok:
            enc_name = f"{c}_target_enc"
            work[enc_name] = _oof_target_encoding(work[c], y_num, N_OOF_FOLDS).values
            added.append(enc_name)
            generated_encoding_cols.add(enc_name)
            encodings.append("target_mean_oof")
            method = f"out-of-fold target mean ({N_OOF_FOLDS}-fold, leakage-safe)"
            used_oof = True

        encoded.append({"column": c, "n_levels": int(n_unique), "encodings": encodings, "method": method})
        # Drop the raw high-cardinality column: one-hot encoding it downstream
        # would explode the feature space (that is exactly why we encoded it).
        work = work.drop(columns=[c])
    if encoded:
        if used_oof:
            notes.append(
                "High-cardinality categoricals were frequency-encoded and target-mean "
                "encoded. Target means use OUT-OF-FOLD K-fold estimates so a row's own "
                "label never enters its own encoding (no leakage). Raw high-card columns "
                "were dropped to avoid a one-hot explosion."
            )
        else:
            notes.append(
                "High-cardinality categoricals were frequency-encoded. Target-mean "
                "encoding was skipped because the target is not numeric or binary."
            )

    # --- Step 3: numeric interactions (capped) ------------------------------
    numeric_cols = [
        c for c in work.columns
        if c != target
        and c not in generated_encoding_cols
        and pd.api.types.is_numeric_dtype(work[c])
    ]
    ranked = _rank_numeric_columns(work, numeric_cols, y_num, y_ok)
    base = ranked[:MAX_INTERACTION_BASE]
    pairs_created = []
    if len(base) >= 2:
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                if len(pairs_created) >= MAX_INTERACTION_COLS:
                    break
                a, b = base[i], base[j]
                name = f"{a}_x_{b}"
                if name in work.columns:
                    continue
                work[name] = (work[a].astype(float) * work[b].astype(float)).values
                added.append(name)
                pairs_created.append((a, b))
            if len(pairs_created) >= MAX_INTERACTION_COLS:
                break
        notes.append(
            f"Created {len(pairs_created)} pairwise product feature(s) from the top "
            f"{len(base)} numeric columns (capped at {MAX_INTERACTION_BASE} base columns "
            f"and {MAX_INTERACTION_COLS} interaction columns to prevent explosion)."
        )

    # --- Step 4: pruning ----------------------------------------------------
    # 4a. constant columns carry no signal
    const_cols = [
        c for c in work.columns
        if c != target and work[c].nunique(dropna=False) <= 1
    ]
    for c in const_cols:
        work = work.drop(columns=[c])
        dropped.append({"column": c, "reason": "constant column (single value, no signal)"})

    # 4b. near-perfectly-correlated numeric columns: keep one of each pair
    num_now = [
        c for c in work.columns
        if c != target and pd.api.types.is_numeric_dtype(work[c])
    ]
    if len(num_now) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = work[num_now].corr().abs()
        to_drop: set = set()
        for i, a in enumerate(num_now):
            if a in to_drop:
                continue
            for b in num_now[i + 1:]:
                if b in to_drop:
                    continue
                r = corr.loc[a, b]
                if pd.notna(r) and r > CORR_DROP_THRESHOLD:
                    to_drop.add(b)
                    dropped.append({
                        "column": b,
                        "reason": f"near-duplicate of '{a}' (|r|={float(r):.3f} > {CORR_DROP_THRESHOLD})",
                    })
        if to_drop:
            work = work.drop(columns=list(to_drop))
    if const_cols or dropped:
        notes.append(
            "Pruned constant columns and one column from each near-perfectly-correlated "
            f"numeric pair (|r| > {CORR_DROP_THRESHOLD}); such columns add noise and "
            "multicollinearity without new signal."
        )

    # keep any surviving added columns list in sync (some added cols may have
    # been pruned again); report 'added' as what was created, 'dropped' as what
    # was removed, which is the honest accounting.
    n_features_after = work.shape[1] - 1  # exclude the target

    report = {
        "added": added,
        "dropped": dropped,
        "encoded": encoded,
        "datetime_expanded": datetime_expanded,
        "n_features_before": n_features_before,
        "n_features_after": n_features_after,
        "notes": notes,
    }

    if show:
        _print_report(target, report)

    return work, report


def _print_report(target: str, report: dict) -> None:
    print(f"\nBreezeML Automatic Feature Engineering - target: '{target}'")
    print("-" * 64)
    print(f"  Features: {report['n_features_before']} in  ->  {report['n_features_after']} out")

    if report["datetime_expanded"]:
        print("\n  Datetime expanded:")
        for item in report["datetime_expanded"]:
            print(f"    - '{item['column']}' -> {', '.join(item['into'])}")

    if report["encoded"]:
        print("\n  High-cardinality encoded:")
        for item in report["encoded"]:
            print(
                f"    - '{item['column']}' ({item['n_levels']} levels): "
                f"{', '.join(item['encodings'])}  [{item['method']}]"
            )

    interaction_added = [a for a in report["added"] if "_x_" in a]
    if interaction_added:
        print("\n  Numeric interactions added:")
        for name in interaction_added:
            print(f"    - {name}")

    if report["dropped"]:
        print("\n  Dropped:")
        for item in report["dropped"]:
            print(f"    - '{item['column']}': {item['reason']}")

    if report["notes"]:
        print("\n  Why:")
        for note in report["notes"]:
            print(f"    * {note}")
    print("-" * 64)
    print(
        f"  Done. {len(report['added'])} column(s) added, "
        f"{len(report['dropped'])} pruned. Result is model-ready.\n"
    )
