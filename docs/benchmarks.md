# Benchmarks: BreezeML vs PyCaret vs LazyPredict

Honest, reproducible numbers. Run them yourself:

```bash
python benchmarks/run_benchmarks.py
```

**Task:** produce a full model leaderboard on sklearn's Wine dataset
(178 rows, 13 features, 3 classes).

**Environment:** Windows 11, Python 3.11, measured 2026-07-05. All three
libraries ran in the same virtual environment on the same machine. Import
times are cold-start medians of 3 runs in fresh interpreters.

## Results

| | BreezeML | PyCaret | LazyPredict |
|---|---|---|---|
| Fresh install (no pip cache) | **2m 36s / 274 MB** | 6m 21s / 952 MB* | * |
| Cold import time | **3.1s** | 6.9s | 7.2s |
| Leaderboard time (Wine) | 9.0s | 19.3s | **3.7s** |
| Best holdout accuracy | **1.000** | 0.984 | **1.000** |
| User lines of code | **3** | 5 | 8 |
| Produces deployable pipeline | **yes** | yes | no |
| Zero lock-in export | **yes** | no | no |

\* PyCaret and LazyPredict were installed together into one venv; PyCaret
accounts for the overwhelming majority of the size and time. During the
benchmark run PyCaret also wrote an unrequested `logs.log` into the working
directory.

Two footnotes in fairness:

- LazyPredict's leaderboard is genuinely the fastest. It also returns no
  usable pipeline: no preprocessing, no persistence, no deployment path.
- BreezeML's leaderboard in this run benchmarked 14 models (XGBoost and
  LightGBM auto-detected in the venv); PyCaret compared a similar set.

## What each column means

- **Install** - `pip install <library>` time into a fresh venv, and the
  resulting site-packages footprint.
- **Import time** - cold `import` in a fresh interpreter (median of 3).
  You pay this every script start, every container cold-start.
- **Leaderboard time** - time to train and rank all built-in models.
- **Best accuracy** - best holdout accuracy on the leaderboard.
- **User LOC** - non-empty lines of user code needed (snippets in
  [`benchmarks/run_benchmarks.py`](https://github.com/venomez-viper/breezeml/blob/main/benchmarks/run_benchmarks.py)).

## What we do NOT claim

- Wine is a small, clean dataset. On large, messy tabular problems,
  AutoGluon or a tuned PyCaret stack may find more accurate ensembles.
- LazyPredict is faster to a first leaderboard on some datasets; it also
  stops there - no pipelines, tuning, persistence, or deployment.
- Accuracy on a 36-row test split has high variance; treat small
  differences as noise.

BreezeML's claim is not "most accurate AutoML." It is: **the fastest path
from DataFrame to a sound, explained, deployable model, with 4 dependencies
and zero lock-in.**
