"""Standard linear classifier baseline (StandardScaler + LogisticRegression)."""

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
                ("scaler", StandardScaler()),
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
