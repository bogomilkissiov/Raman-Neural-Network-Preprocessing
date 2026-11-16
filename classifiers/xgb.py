"""Standard XGBoost gradient-boosted tree baseline for spectral classification."""

from typing import Optional
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class XGBSpectralClassifier:
    """Non-linear tabular baseline (XGBoost) with no 1D spatial inductive bias."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: Optional[int] = 30,
        device: str = "cpu",
        n_jobs: int = 8,
        random_state: int = 42,
    ) -> None:
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for XGBSpectralClassifier. "
                "Install it with `pip install xgboost`."
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model: Optional[XGBClassifier] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBSpectralClassifier":
        """Fit XGBoost classifier with optional validation early stopping."""
        use_es = X_val is not None and y_val is not None and self.early_stopping_rounds
        n_classes = int(max(np.max(y_train), 0) + 1)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

        kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            device=self.device,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
        )
        if use_es:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = XGBClassifier(**kwargs)
        if use_es:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for spectra X of shape (N, L)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for spectra X of shape (N, L)."""
        return self.model.predict_proba(X)
