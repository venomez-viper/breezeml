"""
BreezeML recommenders: collaborative filtering on the core dependencies.

Turns a long-format interactions table (one row per user-item event) into a
user-item matrix, factorizes it with a truncated SVD, and ranks the items a
user has NOT yet touched by their reconstructed score. Implicit feedback
(interaction = 1) and explicit ratings are both supported. Cold-start users
fall back to global popularity, and that fallback is reported honestly instead
of being disguised as personalization.

    from breezeml import recommend

    rec = recommend.collaborative_filter(df, "user", "item", rating_col="stars")
    rec.recommend("u_7", k=5)          # -> [(item, score), ...]
    rec.recommend_report("u_7", k=5)   # -> dict with cold_start/method flags

When NOT to use it
------------------
- This is pure collaborative filtering. It knows nothing about item or user
  side features (genre, price, text, demographics). If your signal lives in
  content, use a content-based or hybrid model instead.
- Cold start is popularity-only. A brand-new user or item has no factors, so
  everyone gets the same globally popular list until they interact. If most of
  your traffic is new users, this will feel generic.
- It needs enough interactions per user to personalize. Users or items with a
  single interaction cannot be placed in latent space with any confidence; the
  fit warns about them but cannot invent history that is not there.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

__all__ = ["Recommender", "collaborative_filter"]

SPARSE_DENSITY = 0.01  # below this the matrix is "extremely sparse"


class Recommender:
    """Collaborative-filtering recommender backed by a truncated SVD.

    Parameters
    ----------
    n_components : int
        Number of latent factors to keep. Clipped to the matrix rank
        (``min(n_users, n_items) - 1``) at fit time so the SVD never asks
        for more components than the data can support.
    random_state : int
        Seed for the randomized SVD solver.
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        rating_col: str | None = None,
        show: bool = True,
    ) -> "Recommender":
        """Build the user-item matrix and factorize it.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format interactions, one row per user-item event.
        user_col, item_col : str
            Column names identifying the user and the item.
        rating_col : str or None
            Explicit rating column. If None, feedback is treated as implicit
            (1 for any observed interaction).
        show : bool
            Print the honesty report at fit time (default True).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        for col in (user_col, item_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available columns: {list(df.columns)}"
                )
        if rating_col is not None and rating_col not in df.columns:
            raise ValueError(
                f"rating_col '{rating_col}' not found. Available columns: {list(df.columns)}"
            )

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.implicit = rating_col is None

        work = df[[user_col, item_col] + ([rating_col] if rating_col else [])].copy()
        work = work.dropna(subset=[user_col, item_col])
        if len(work) == 0:
            raise ValueError("No usable interactions after dropping missing user/item values.")

        if self.implicit:
            values = pd.Series(1.0, index=work.index)
            agg = "max"
        else:
            values = work[rating_col].astype(float)
            agg = "mean"
        work = work.assign(_value=values)

        # Duplicate user-item pairs are collapsed: max for implicit (presence),
        # mean for explicit ratings.
        matrix = work.pivot_table(
            index=user_col, columns=item_col, values="_value", aggfunc=agg
        )

        self.users_ = list(matrix.index)
        self.items_ = list(matrix.columns)
        self._user_pos = {u: i for i, u in enumerate(self.users_)}
        n_users, n_items = matrix.shape

        # Where a user actually interacted (before filling gaps with zeros).
        self.interacted_mask_ = matrix.notna().to_numpy()
        filled = matrix.fillna(0.0).to_numpy(dtype=float)
        self.matrix_ = filled

        # Item popularity = raw interaction counts, for the cold-start fallback.
        pop = work.groupby(item_col).size()
        self.item_popularity_ = pop.reindex(self.items_).fillna(0).astype(int)
        self._popular_order = list(
            self.item_popularity_.sort_values(ascending=False).index
        )

        n_interactions = int(self.interacted_mask_.sum())
        density = n_interactions / float(n_users * n_items) if n_users and n_items else 0.0

        # Clip components to the achievable rank of the matrix.
        max_rank = max(min(n_users, n_items) - 1, 1)
        n_comp = int(min(self.n_components, max_rank))
        self.n_components_used_ = n_comp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            user_factors = svd.fit_transform(filled)
        self.svd_ = svd
        # Reconstructed approximation of the full user-item matrix.
        self.reconstruction_ = user_factors @ svd.components_

        users_single = int((self.interacted_mask_.sum(axis=1) == 1).sum())
        items_single = int((self.interacted_mask_.sum(axis=0) == 1).sum())

        self.stats = {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_interactions": n_interactions,
            "shape": (int(n_users), int(n_items)),
            "density": round(float(density), 6),
            "n_components_used": n_comp,
            "implicit": bool(self.implicit),
            "users_with_one_interaction": users_single,
            "items_with_one_interaction": items_single,
        }

        self.fitted_ = True
        if show:
            self._print_report()
        return self

    def _print_report(self) -> None:
        s = self.stats
        mode = "implicit (presence = 1)" if self.implicit else f"explicit ('{self.rating_col}')"
        print(f"\nBreezeML Recommender - {mode}")
        print("-" * 64)
        print(
            f"  Matrix: {s['n_users']:,} users x {s['n_items']:,} items, "
            f"{s['n_interactions']:,} interactions, density {s['density']:.2%}"
        )
        print(f"  Latent factors: {s['n_components_used']} (SVD, clipped to matrix rank)")
        if s["density"] < SPARSE_DENSITY:
            print(
                f"  ! [WARNING] Matrix is extremely sparse (density {s['density']:.3%} < "
                f"{SPARSE_DENSITY:.0%}). Latent factors will be noisy; recommendations may "
                "be little better than popularity."
            )
        if s["users_with_one_interaction"] > 0:
            print(
                f"  ! [WARNING] {s['users_with_one_interaction']:,} user(s) have a single "
                "interaction. They cannot be personalized reliably."
            )
        if s["items_with_one_interaction"] > 0:
            print(
                f"  ! [WARNING] {s['items_with_one_interaction']:,} item(s) have a single "
                "interaction. Their factors rest on one data point."
            )
        print("-" * 64 + "\n")

    # ------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Recommender is not fitted yet. Call fit() first.")

    def _score_known_user(self, user) -> np.ndarray:
        """Predicted scores for every item, with already-seen items masked out."""
        pos = self._user_pos[user]
        scores = self.reconstruction_[pos].copy()
        scores[self.interacted_mask_[pos]] = -np.inf  # never re-recommend seen items
        return scores

    def _popular_items(self, k: int, exclude: set | None = None) -> list:
        exclude = exclude or set()
        out = [it for it in self._popular_order if it not in exclude]
        return out[:k]

    def recommend(self, user, k: int = 5) -> list:
        """Top-``k`` recommendations for ``user`` as ``[(item, score), ...]``.

        Known users are ranked by reconstructed SVD score over items they have
        not interacted with. Unknown users fall back to global popularity (see
        ``recommend_report`` for the cold-start flag).
        """
        self._check_fitted()
        if k <= 0:
            return []
        if user not in self._user_pos:
            popular = self._popular_items(k)
            return [(it, float(self.item_popularity_[it])) for it in popular]

        scores = self._score_known_user(user)
        # Rank items by score; -inf entries (already seen) sink to the bottom.
        order = np.argsort(scores)[::-1]
        results = []
        for idx in order:
            if len(results) >= k:
                break
            if not np.isfinite(scores[idx]):
                continue  # all remaining items were already interacted with
            results.append((self.items_[idx], float(scores[idx])))

        if len(results) < k:
            # Not enough unseen items with finite scores; pad with popularity.
            already = {it for it, _ in results}
            seen_pos = self.interacted_mask_[self._user_pos[user]]
            seen_items = {self.items_[i] for i in np.where(seen_pos)[0]}
            for it in self._popular_items(k, exclude=already | seen_items):
                if len(results) >= k:
                    break
                results.append((it, float(self.item_popularity_[it])))
        return results

    def recommend_report(self, user, k: int = 5) -> dict:
        """Recommendations plus honest cold-start / method metadata.

        Returns
        -------
        dict
            ``{"recommendations": [(item, score), ...], "cold_start": bool,
            "method": "svd" | "popularity", "user": user, "k": k, "note": str}``
        """
        self._check_fitted()
        cold = user not in self._user_pos
        recs = self.recommend(user, k=k)
        if cold:
            method = "popularity"
            note = (
                "Cold start: this user was not seen in training, so recommendations "
                "are the globally most popular items, not personalized. Scores are "
                "interaction counts."
            )
        else:
            method = "svd"
            note = (
                f"Personalized via truncated SVD ({self.n_components_used_} factors). "
                "Scores are reconstructed affinities, not probabilities."
            )
        return {
            "recommendations": recs,
            "cold_start": bool(cold),
            "method": method,
            "user": user,
            "k": int(k),
            "note": note,
        }


def collaborative_filter(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str | None = None,
    n_components: int = 20,
    show: bool = True,
) -> Recommender:
    """Fit and return a :class:`Recommender` in one call.

    Convenience wrapper: builds the user-item matrix from ``df``, factorizes it
    with a truncated SVD, and returns the fitted model ready for
    ``recommend`` / ``recommend_report``.
    """
    rec = Recommender(n_components=n_components)
    rec.fit(df, user_col, item_col, rating_col=rating_col, show=show)
    return rec
