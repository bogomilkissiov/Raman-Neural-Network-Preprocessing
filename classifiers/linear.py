"""Linear classifier baseline (global [0, 1] range scaling + LogisticRegression)."""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class GlobalRangeScaler(BaseEstimator, TransformerMixin):
    """Scale all spectra by a single global (min, max) from training data.

    Preserves relative peak heights across wavenumbers and amplitude fidelity
    across spectra, avoiding per-channel variance distortion.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GlobalRangeScaler":
        del y
        arr = np.asarray(X, dtype=np.float64)
        self.data_min_ = float(np.min(arr))
        data_max = float(np.max(arr))
        span = data_max - self.data_min_
        self.data_scale_ = span if span > 0.0 else 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return (arr - self.data_min_) / self.data_scale_


class LinearSpectralClassifier:
    """Minimal-capacity linear baseline for downstream Raman classification."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        solver: str = "lbfgs",
        random_state: int = 42,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pipeline = Pipeline(
            [
                ("scaler", GlobalRangeScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        solver=self.solver,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LinearSpectralClassifier":
        """Fit scaler + logistic regression on training spectra (X_val unused)."""
        del X_val, y_val
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for spectra X of shape (N, L)."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for spectra X of shape (N, L)."""
        return self.pipeline.predict_proba(X)
