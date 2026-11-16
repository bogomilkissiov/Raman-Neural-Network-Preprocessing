"""Shared downstream classification benchmark runner for Raman preprocessing.

Turns running any preprocessing pipeline (Raw, Conventional, PolyGaussNet V3,
PolyGaussNet V3.4-8sec, etc.) through the classifier suite (Linear baseline,
XGBoost, and Bacteria-ID 1D ResNet) into a single reproducible function call.
"""

from __future__ import annotations

import csv
import datetime
import gc
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

from .linear import LinearSpectralClassifier
from .resnet1d import BacteriaResNetClassifier
from .xgb import XGBClassifier as _XGB_BACKEND
from .xgb import XGBSpectralClassifier


DEFAULT_CLASSIFIER_ORDER: Tuple[str, ...] = ("Linear", "XGBoost", "BacteriaResNet")

_CLASSIFIER_ALIASES: Dict[str, str] = {
    "linear": "Linear",
    "linearspectralclassifier": "Linear",
    "logreg": "Linear",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "xgbspectralclassifier": "XGBoost",
    "resnet": "BacteriaResNet",
    "resnet1d": "BacteriaResNet",
    "bacteria": "BacteriaResNet",
    "bacteriaresnet": "BacteriaResNet",
    "bacteria_resnet": "BacteriaResNet",
    "bacteria-id": "BacteriaResNet",
    "bacteriaresnetclassifier": "BacteriaResNet",
}

_METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "roc_auc_ovr",
    "fit_time_sec",
    "predict_time_sec",
)


def default_n_jobs() -> int:
    """Return Slurm-aware CPU thread count."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return min(os.cpu_count() or 4, 8)


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve PyTorch compute device cleanly across OSCAR CUDA and local CPU/MPS."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _canonical_classifier_name(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "").replace("-", "_")
    return _CLASSIFIER_ALIASES.get(cleaned, name)


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split unsplit `(X, y)` into stratified train/val/test sets."""
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not (0.0 <= val_size < 1.0 - test_size):
        raise ValueError(f"val_size ({val_size}) must satisfy 0 <= val_size < 1 - test_size.")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_state,
    )
    if val_size == 0.0:
        empty_x = X_train_val[:0]
        empty_y = y_train_val[:0]
        return X_train_val, y_train_val, empty_x, empty_y, X_test, y_test

    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=rel_val,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def encode_labels(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Map arbitrary class labels into contiguous `0..C-1` integers without leakage."""
    y_tr_raw = np.asarray(y_train)
    encoder = LabelEncoder()
    y_tr_enc = encoder.fit_transform(y_tr_raw).astype(np.int64)
    known = set(encoder.classes_.tolist())

    def _transform(arr: Optional[np.ndarray], split_name: str) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr_np = np.asarray(arr)
        unseen = set(arr_np.tolist()) - known
        if unseen:
            raise ValueError(
                f"{split_name} contains classes not present in y_train: {sorted(unseen)[:5]}"
            )
        return encoder.transform(arr_np).astype(np.int64)

    y_val_enc = _transform(y_val, "y_val")
    y_te_enc = _transform(y_test, "y_test")
    class_names = [str(c) for c in encoder.classes_]
    return y_tr_enc, y_val_enc, y_te_enc, class_names


def _validate_2d_spectra(X: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (N, L) for {name}, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad_count = int(np.size(arr) - np.sum(np.isfinite(arr)))
        raise ValueError(f"{name} contains {bad_count} non-finite (NaN/Inf) values.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _apply_preprocessor(
    method_name: str,
    preprocessor: Any,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Run a preprocessing callable (or unpack precomputed arrays) with timing."""
    timings: Dict[str, Any] = {
        "train": 0.0,
        "val": 0.0,
        "test": 0.0,
        "total": 0.0,
        "mode": "identity",
    }

    if preprocessor is None:
        timings["mode"] = "identity"
        X_tr_p = _validate_2d_spectra(X_train, f"{method_name}:train")
        X_val_p = (
            _validate_2d_spectra(X_val, f"{method_name}:val")
            if X_val is not None
            else None
        )
        X_te_p = _validate_2d_spectra(X_test, f"{method_name}:test")
    elif callable(preprocessor):
        timings["mode"] = "timed"
        t0 = time.perf_counter()
        X_tr_raw = preprocessor(_validate_2d_spectra(X_train, "X_train"))
        timings["train"] = time.perf_counter() - t0
        X_tr_p = _validate_2d_spectra(X_tr_raw, f"{method_name}:train")

        if X_val is not None:
            t1 = time.perf_counter()
            X_val_raw = preprocessor(_validate_2d_spectra(X_val, "X_val"))
            timings["val"] = time.perf_counter() - t1
            X_val_p = _validate_2d_spectra(X_val_raw, f"{method_name}:val")
        else:
            X_val_p = None

        t2 = time.perf_counter()
        X_te_raw = preprocessor(_validate_2d_spectra(X_test, "X_test"))
        timings["test"] = time.perf_counter() - t2
        X_te_p = _validate_2d_spectra(X_te_raw, f"{method_name}:test")
    elif isinstance(preprocessor, Mapping):
        timings["mode"] = "precomputed"
        if "train" not in preprocessor or "test" not in preprocessor:
            raise KeyError(
                f"Precomputed mapping for '{method_name}' must contain 'train' and 'test'."
            )
        X_tr_p = _validate_2d_spectra(preprocessor["train"], f"{method_name}:train")
        X_val_raw = preprocessor.get("val")
        X_val_p = (
            _validate_2d_spectra(X_val_raw, f"{method_name}:val")
            if X_val_raw is not None
            else None
        )
        X_te_p = _validate_2d_spectra(preprocessor["test"], f"{method_name}:test")
    elif isinstance(preprocessor, (tuple, list)):
        timings["mode"] = "precomputed"
        if len(preprocessor) == 3:
            X_tr_p = _validate_2d_spectra(preprocessor[0], f"{method_name}:train")
            X_val_p = (
                _validate_2d_spectra(preprocessor[1], f"{method_name}:val")
                if preprocessor[1] is not None
                else None
            )
            X_te_p = _validate_2d_spectra(preprocessor[2], f"{method_name}:test")
        elif len(preprocessor) == 2:
            X_tr_p = _validate_2d_spectra(preprocessor[0], f"{method_name}:train")
            X_val_p = None
            X_te_p = _validate_2d_spectra(preprocessor[1], f"{method_name}:test")
        else:
            raise ValueError(
                f"Precomputed tuple for '{method_name}' must have length 2 or 3, got {len(preprocessor)}."
            )
    else:
        raise TypeError(
            f"Unsupported preprocessor for '{method_name}': expected None, callable, "
            f"tuple/list of splits, or dict; got {type(preprocessor).__name__}."
        )

    if X_tr_p.shape[0] != X_train.shape[0]:
        raise ValueError(
            f"{method_name}: preprocessed train count ({X_tr_p.shape[0]}) != input ({X_train.shape[0]})."
        )
    if X_val is not None and X_val_p is not None and X_val_p.shape[0] != X_val.shape[0]:
        raise ValueError(
            f"{method_name}: preprocessed val count ({X_val_p.shape[0]}) != input ({X_val.shape[0]})."
        )
    if X_te_p.shape[0] != X_test.shape[0]:
        raise ValueError(
            f"{method_name}: preprocessed test count ({X_te_p.shape[0]}) != input ({X_test.shape[0]})."
        )
    if X_tr_p.shape[1] != X_te_p.shape[1] or (
        X_val_p is not None and X_tr_p.shape[1] != X_val_p.shape[1]
    ):
        raise ValueError(
            f"{method_name}: inconsistent feature dimensions across splits "
            f"(train={X_tr_p.shape[1]}, val={X_val_p.shape[1] if X_val_p is not None else None}, test={X_te_p.shape[1]})."
        )

    timings["total"] = timings["train"] + timings["val"] + timings["test"]
    n_total = X_tr_p.shape[0] + (X_val_p.shape[0] if X_val_p is not None else 0) + X_te_p.shape[0]
    timings["spectra_per_sec"] = (
        float(n_total / timings["total"]) if timings["total"] > 0 else float("inf")
    )
    return X_tr_p, X_val_p, X_te_p, timings


def make_default_classifiers(
    random_state: int = 42,
    include_classifiers: Optional[Sequence[str]] = None,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 7,
    lr: float = 1e-3,
    in_channels: int = 64,
    hidden_sizes: Sequence[int] = (100, 100, 100, 100, 100, 100),
    num_blocks: Sequence[int] = (2, 2, 2, 2, 2, 2),
    device: Optional[str] = None,
    xgb_n_estimators: int = 500,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_early_stopping_rounds: Optional[int] = 30,
    xgb_device: str = "cpu",
    xgb_n_jobs: Optional[int] = None,
    linear_C: float = 1.0,
    linear_max_iter: int = 2000,
    linear_kwargs: Optional[Mapping[str, Any]] = None,
    xgb_kwargs: Optional[Mapping[str, Any]] = None,
    resnet_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Instantiate the default 3-family classifier suite with fair fixed hyperparameters."""
    if include_classifiers is None:
        selected = list(DEFAULT_CLASSIFIER_ORDER)
    else:
        selected = [_canonical_classifier_name(c) for c in include_classifiers]

    resolved_device = resolve_device(device)
    resolved_jobs = default_n_jobs() if xgb_n_jobs is None else xgb_n_jobs

    lin_cfg: Dict[str, Any] = {
        "C": linear_C,
        "max_iter": linear_max_iter,
        "random_state": random_state,
    }
    if linear_kwargs:
        lin_cfg.update(linear_kwargs)

    xgb_cfg: Dict[str, Any] = {
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": xgb_learning_rate,
        "early_stopping_rounds": xgb_early_stopping_rounds,
        "device": xgb_device,
        "n_jobs": resolved_jobs,
        "random_state": random_state,
    }
    if xgb_kwargs:
        xgb_cfg.update(xgb_kwargs)

    res_cfg: Dict[str, Any] = {
        "hidden_sizes": hidden_sizes,
        "num_blocks": num_blocks,
        "in_channels": in_channels,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "device": resolved_device,
        "random_state": random_state,
    }
    if resnet_kwargs:
        res_cfg.update(resnet_kwargs)

    builders: Dict[str, Callable[[], Any]] = {
        "Linear": lambda: LinearSpectralClassifier(**lin_cfg),
        "XGBoost": lambda: XGBSpectralClassifier(**xgb_cfg),
        "BacteriaResNet": lambda: BacteriaResNetClassifier(**res_cfg),
    }

    instances: Dict[str, Any] = {}
    for name in selected:
        if name not in builders:
            raise ValueError(
                f"Unknown classifier '{name}'. Supported defaults: {list(builders.keys())}."
            )
        if name == "XGBoost" and _XGB_BACKEND is None:
            raise ImportError(
                "xgboost is required for 'XGBoost'. Install it via `pip install xgboost` "
                "(e.g., on OSCAR: `pip install xgboost`), or pass "
                "include_classifiers=('Linear', 'BacteriaResNet') to skip it."
            )
        instances[name] = builders[name]()
    return instances


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute reviewer-standard classification metrics on fixed ground-truth labels."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    present_labels = np.unique(y_t)
    inferred_classes = int(
        max(
            int(np.max(y_t)) + 1 if y_t.size else 0,
            int(np.max(y_p)) + 1 if y_p.size else 0,
            y_prob.shape[1] if y_prob is not None and y_prob.ndim == 2 else 0,
        )
    )
    total_classes = int(n_classes) if n_classes is not None else inferred_classes
    all_labels = list(range(total_classes))

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(
        f1_score(y_t, y_p, labels=present_labels, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_t, y_p, labels=present_labels, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_t, y_p, labels=all_labels).tolist()

    ll = float("nan")
    auc = float("nan")
    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=np.float64)
        if probs.ndim == 2 and probs.shape[1] == total_classes:
            row_sums = probs.sum(axis=1, keepdims=True)
            valid_rows = row_sums[:, 0] > 0
            if np.all(valid_rows):
                probs_norm = probs / row_sums
                ll = float(log_loss(y_t, probs_norm, labels=all_labels))
                try:
                    if total_classes == 2 and len(present_labels) == 2:
                        auc = float(roc_auc_score(y_t, probs_norm[:, 1]))
                    elif total_classes > 2 and len(present_labels) == total_classes:
                        auc = float(
                            roc_auc_score(
                                y_t,
                                probs_norm,
                                multi_class="ovr",
                                average="macro",
                                labels=all_labels,
                            )
                        )
                except ValueError:
                    auc = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": ll,
        "roc_auc_ovr": auc,
        "confusion_matrix": cm,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_method: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> Dict[str, Any]:
    """Run exact two-sided McNemar's test comparing two models on the same test set."""
    y_t = np.asarray(y_true)
    correct_m = np.asarray(y_pred_method) == y_t
    correct_b = np.asarray(y_pred_baseline) == y_t

    both_right = int(np.sum(correct_m & correct_b))
    method_only = int(np.sum(correct_m & ~correct_b))   # n10: method wins, baseline fails
    baseline_only = int(np.sum(~correct_m & correct_b)) # n01: baseline wins, method fails
    both_wrong = int(np.sum(~correct_m & ~correct_b))

    discordant = method_only + baseline_only
    if discordant == 0:
        pvalue = 1.0
    elif hasattr(scipy.stats, "binomtest"):
        pvalue = float(
            scipy.stats.binomtest(
                method_only, discordant, p=0.5, alternative="two-sided"
            ).pvalue
        )
    else:  # SciPy < 1.7 fallback
        pvalue = float(scipy.stats.binom_test(method_only, discordant, p=0.5))

    return {
        "both_correct": both_right,
        "method_only_correct": method_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "p_value": pvalue,
    }


def _aggregate_seed_runs(seed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_seeds": len(seed_runs)}
    for key in _METRIC_KEYS:
        vals = np.array([float(r[key]) for r in seed_runs], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(np.mean(finite))
            summary[f"{key}_std"] = float(np.std(finite, ddof=0))
    return summary


def evaluate_method(
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    preprocessor: Any = None,
    seeds: Sequence[int] = (42,),
    n_classes: Optional[int] = None,
    classifier_factories: Optional[Mapping[str, Callable[[int], Any]]] = None,
    include_classifiers: Optional[Sequence[str]] = None,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 7,
    lr: float = 1e-3,
    device: Optional[str] = None,
    xgb_n_estimators: int = 500,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_early_stopping_rounds: Optional[int] = 30,
    xgb_device: str = "cpu",
    xgb_n_jobs: Optional[int] = None,
    linear_kwargs: Optional[Mapping[str, Any]] = None,
    xgb_kwargs: Optional[Mapping[str, Any]] = None,
    resnet_kwargs: Optional[Mapping[str, Any]] = None,
    return_models: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Preprocess `(X_train, X_val, X_test)` once and train/evaluate all classifiers.

    Supports repeated evaluation across `seeds` under identical epochs and splits.
    If string/non-contiguous labels are passed directly here, encodes them cleanly.
    """
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one integer.")

    y_tr_enc, y_val_enc, y_te_enc, class_names = encode_labels(y_train, y_val, y_test)
    total_classes = int(n_classes) if n_classes is not None else len(class_names)

    if verbose:
        print(f"\n[Benchmark] Preprocessing method: '{method_name}' ...", flush=True)

    X_tr_p, X_val_p, X_te_p, prep_timings = _apply_preprocessor(
        method_name, preprocessor, X_train, X_val, X_test
    )
    if X_val_p is not None and y_val_enc is not None and X_val_p.shape[0] != y_val_enc.shape[0]:
        raise ValueError(
            f"{method_name}: preprocessed val count ({X_val_p.shape[0]}) != y_val ({y_val_enc.shape[0]})."
        )

    if verbose:
        print(
            f"  -> Shapes: train={X_tr_p.shape}, "
            f"val={X_val_p.shape if X_val_p is not None else None}, test={X_te_p.shape} | "
            f"prep time={prep_timings['total']:.2f}s "
            f"({prep_timings['spectra_per_sec']:.1f} spec/s)",
            flush=True,
        )

    seed_list = [int(s) for s in seeds]

    def _build_for_seed(s: int) -> Dict[str, Any]:
        if classifier_factories is not None:
            built: Dict[str, Any] = {}
            for cname, factory in classifier_factories.items():
                if include_classifiers is not None:
                    allowed = {_canonical_classifier_name(x) for x in include_classifiers}
                    if _canonical_classifier_name(cname) not in allowed:
                        continue
                built[cname] = factory(s)
            return built
        return make_default_classifiers(
            random_state=s,
            include_classifiers=include_classifiers,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            lr=lr,
            device=device,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
            xgb_early_stopping_rounds=xgb_early_stopping_rounds,
            xgb_device=xgb_device,
            xgb_n_jobs=xgb_n_jobs,
            linear_kwargs=linear_kwargs,
            xgb_kwargs=xgb_kwargs,
            resnet_kwargs=resnet_kwargs,
        )

    probe = _build_for_seed(seed_list[0])
    classifier_names = list(probe.keys())
    del probe

    classifiers_out: Dict[str, Dict[str, Any]] = {
        cname: {"per_seed": [], "seed_predictions": {}, "models": {}}
        for cname in classifier_names
    }

    for seed in seed_list:
        models_for_seed = _build_for_seed(seed)
        for cname, clf in models_for_seed.items():
            t_fit0 = time.perf_counter()
            clf.fit(X_tr_p, y_tr_enc, X_val=X_val_p, y_val=y_val_enc)
            fit_time = time.perf_counter() - t_fit0

            t_pred0 = time.perf_counter()
            y_prob = clf.predict_proba(X_te_p) if hasattr(clf, "predict_proba") else None
            y_pred = np.asarray(clf.predict(X_te_p), dtype=np.int64)
            pred_time = time.perf_counter() - t_pred0

            metrics = compute_classification_metrics(
                y_te_enc, y_pred, y_prob=y_prob, n_classes=total_classes
            )
            metrics["seed"] = seed
            metrics["fit_time_sec"] = float(fit_time)
            metrics["predict_time_sec"] = float(pred_time)
            classifiers_out[cname]["per_seed"].append(metrics)
            classifiers_out[cname]["seed_predictions"][seed] = y_pred

            if return_models:
                classifiers_out[cname]["models"][seed] = clf
            else:
                del clf

            if verbose:
                print(
                    f"  -> [{cname:14s} | seed={seed}] "
                    f"Acc={metrics['accuracy'] * 100:6.2f}% | "
                    f"BalAcc={metrics['balanced_accuracy'] * 100:6.2f}% | "
                    f"MacroF1={metrics['macro_f1'] * 100:6.2f}% | "
                    f"Fit={fit_time:6.2f}s",
                    flush=True,
                )

            # Release PyTorch GPU allocations cleanly across method x seed loops on OSCAR.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for cname in classifier_names:
        classifiers_out[cname]["summary"] = _aggregate_seed_runs(
            classifiers_out[cname]["per_seed"]
        )
        if not return_models:
            classifiers_out[cname].pop("models", None)

    return {
        "method": method_name,
        "input_dim": int(X_tr_p.shape[1]),
        "n_train": int(X_tr_p.shape[0]),
        "n_val": int(X_val_p.shape[0]) if X_val_p is not None else 0,
        "n_test": int(X_te_p.shape[0]),
        "n_classes": total_classes,
        "class_names": class_names,
        "seeds": seed_list,
        "preprocessing_time_sec": prep_timings,
        "classifiers": classifiers_out,
    }


def add_baseline_comparisons(
    method_results: Mapping[str, Dict[str, Any]],
    y_test_encoded: np.ndarray,
    baseline_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach paired metric deltas and exact McNemar p-values relative to a baseline."""
    method_names = list(method_results.keys())
    if not method_names:
        return {"baseline_method": None, "comparisons": {}}

    if baseline_method is None:
        base_name = "Raw" if "Raw" in method_results else method_names[0]
    else:
        if baseline_method not in method_results:
            raise KeyError(
                f"baseline_method '{baseline_method}' not found in evaluated methods {method_names}."
            )
        base_name = baseline_method

    base_res = method_results[base_name]
    comparisons: Dict[str, Dict[str, Any]] = {}

    for mname, mres in method_results.items():
        comparisons[mname] = {}
        for cname, cdata in mres["classifiers"].items():
            base_cdata = base_res["classifiers"].get(cname)
            if base_cdata is None:
                continue

            m_sum = cdata["summary"]
            b_sum = base_cdata["summary"]
            acc_delta = float(m_sum["accuracy_mean"] - b_sum["accuracy_mean"])
            f1_delta = float(m_sum["macro_f1_mean"] - b_sum["macro_f1_mean"])
            ll_delta = float(m_sum["log_loss_mean"] - b_sum["log_loss_mean"])

            per_seed_mcnemar: List[Dict[str, Any]] = []
            m_preds = cdata.get("seed_predictions", {})
            b_preds = base_cdata.get("seed_predictions", {})
            for seed, y_pm in m_preds.items():
                if seed in b_preds and mname != base_name:
                    test_res = mcnemar_test(y_test_encoded, y_pm, b_preds[seed])
                    test_res["seed"] = seed
                    per_seed_mcnemar.append(test_res)

            pvals = [r["p_value"] for r in per_seed_mcnemar]
            comparisons[mname][cname] = {
                "baseline": base_name,
                "accuracy_delta": acc_delta,
                "macro_f1_delta": f1_delta,
                "log_loss_delta": ll_delta,
                "mcnemar_per_seed": per_seed_mcnemar,
                "mcnemar_p_median": float(np.median(pvals)) if pvals else (1.0 if mname == base_name else float("nan")),
                "mcnemar_p_max": float(np.max(pvals)) if pvals else (1.0 if mname == base_name else float("nan")),
            }

    return {"baseline_method": base_name, "comparisons": comparisons}


def _fmt_pct_pm(mean_val: float, std_val: float, multi_seed: bool) -> str:
    if not np.isfinite(mean_val):
        return "   N/A   "
    if multi_seed and np.isfinite(std_val):
        return f"{mean_val * 100:6.2f}+-{std_val * 100:4.2f}%"
    return f"{mean_val * 100:6.2f}%"


def _fmt_float_pm(mean_val: float, std_val: float, multi_seed: bool) -> str:
    if not np.isfinite(mean_val):
        return "   N/A  "
    if multi_seed and np.isfinite(std_val):
        return f"{mean_val:6.4f}+-{std_val:5.4f}"
    return f"{mean_val:6.4f}"


def format_benchmark_report(benchmark_results: Dict[str, Any]) -> str:
    """Format a reviewer-ready plain-text summary report from benchmark results."""
    meta = benchmark_results["metadata"]
    methods = benchmark_results["methods"]
    comp_block = benchmark_results.get("baseline_comparisons", {})
    base_name = comp_block.get("baseline_method")
    comparisons = comp_block.get("comparisons", {})
    multi_seed = len(meta["seeds"]) > 1

    lines: List[str] = []
    lines.append("=" * 98)
    lines.append("DOWNSTREAM RAMAN CLASSIFICATION BENCHMARK REPORT")
    lines.append("=" * 98)
    lines.append(f"Timestamp:        {meta['timestamp']}")
    lines.append(f"Dataset Name:     {meta.get('dataset_name', 'custom')}")
    lines.append(
        f"Splits (N):       train={meta['n_train']:,} | val={meta['n_val']:,} | test={meta['n_test']:,} "
        f"| classes={meta['n_classes']}"
    )
    lines.append(f"Seeds:            {meta['seeds']}")
    lines.append(f"Baseline Method:  {base_name}")
    lines.append(
        f"Training Config:  ResNet epochs={meta.get('epochs')}, batch_size={meta.get('batch_size')}, "
        f"patience={meta.get('patience')}, lr={meta.get('lr')}, device={meta.get('device')}"
    )
    lines.append("")

    # Section 1: Preprocessing cost
    lines.append("-" * 98)
    lines.append("1. PREPROCESSING FEATURE DIMENSIONS & THROUGHPUT")
    lines.append("-" * 98)
    lines.append(
        f"{'Method':<24} | {'Feat Dim':>8} | {'Train (s)':>9} | {'Val (s)':>8} | {'Test (s)':>8} | {'Throughput':>14}"
    )
    lines.append("-" * 98)
    for mname, mres in methods.items():
        pt = mres["preprocessing_time_sec"]
        mode = pt.get("mode", "timed")
        if mode == "identity":
            tput = "identity (0s)"
        elif mode == "precomputed":
            tput = "precomputed"
        elif np.isfinite(pt["spectra_per_sec"]):
            tput = f"{pt['spectra_per_sec']:,.1f} spec/s"
        else:
            tput = "instant"
        lines.append(
            f"{mname:<24} | {mres['input_dim']:>8d} | {pt['train']:>9.2f} | "
            f"{pt['val']:>8.2f} | {pt['test']:>8.2f} | {tput:>14}"
        )
    lines.append("")

    # Section 2: Downstream classifier accuracy table
    lines.append("-" * 98)
    lines.append("2. DOWNSTREAM TEST CLASSIFICATION METRICS (EQUAL SPLITS, SEEDS & EPOCHS)")
    lines.append("-" * 98)
    header = (
        f"{'Method':<22} | {'Classifier':<14} | {'Accuracy':>14} | {'Bal Acc':>14} | "
        f"{'Macro F1':>14} | {'LogLoss':>14} | {'AUC-OvR':>9} | {'Fit(s)':>7}"
    )
    lines.append(header)
    lines.append("-" * 98)
    for mname, mres in methods.items():
        for cname, cdata in mres["classifiers"].items():
            s = cdata["summary"]
            acc_str = _fmt_pct_pm(s["accuracy_mean"], s["accuracy_std"], multi_seed)
            bal_str = _fmt_pct_pm(
                s["balanced_accuracy_mean"], s["balanced_accuracy_std"], multi_seed
            )
            f1_str = _fmt_pct_pm(s["macro_f1_mean"], s["macro_f1_std"], multi_seed)
            ll_str = _fmt_float_pm(s["log_loss_mean"], s["log_loss_std"], multi_seed)
            auc_val = s["roc_auc_ovr_mean"]
            auc_str = f"{auc_val:7.4f}" if np.isfinite(auc_val) else "   N/A "
            fit_val = s["fit_time_sec_mean"]
            lines.append(
                f"{mname:<22} | {cname:<14} | {acc_str:>14} | {bal_str:>14} | "
                f"{f1_str:>14} | {ll_str:>14} | {auc_str:>9} | {fit_val:>7.2f}"
            )
    lines.append("")

    # Section 3: Paired statistical significance vs baseline
    if base_name is not None and comparisons:
        lines.append("-" * 98)
        lines.append(f"3. PAIRED IMPROVEMENTS & EXACT McNEMAR SIGNIFICANCE VS '{base_name}'")
        lines.append("-" * 98)
        lines.append(
            f"{'Method':<22} | {'Classifier':<14} | {'dAcc (pp)':>10} | {'dMacroF1':>10} | "
            f"{'dLogLoss':>10} | {'McNemar p (med)':>15} | {'Wins/Loss (s0)':>14}"
        )
        lines.append("-" * 98)
        for mname in methods:
            if mname == base_name:
                continue
            for cname, comp in comparisons.get(mname, {}).items():
                d_acc = comp["accuracy_delta"] * 100.0
                d_f1 = comp["macro_f1_delta"] * 100.0
                d_ll = comp["log_loss_delta"]
                pval = comp["mcnemar_p_median"]
                p_str = f"{pval:.2e}" if np.isfinite(pval) else "N/A"
                if comp["mcnemar_per_seed"]:
                    s0 = comp["mcnemar_per_seed"][0]
                    wl_str = f"{s0['method_only_correct']}/{s0['baseline_only_correct']}"
                else:
                    wl_str = "N/A"
                lines.append(
                    f"{mname:<22} | {cname:<14} | {d_acc:>+9.2f}% | {d_f1:>+9.2f}% | "
                    f"{d_ll:>+10.4f} | {p_str:>15} | {wl_str:>14}"
                )
        lines.append("")

    # Section 4: Per-classifier ranking summary
    lines.append("-" * 98)
    lines.append("4. BEST PREPROCESSING METHOD PER CLASSIFIER (BY TEST ACCURACY)")
    lines.append("-" * 98)
    all_classifiers: List[str] = []
    for mres in methods.values():
        for cname in mres["classifiers"]:
            if cname not in all_classifiers:
                all_classifiers.append(cname)
    for cname in all_classifiers:
        ranked = [
            (mname, mres["classifiers"][cname]["summary"]["accuracy_mean"])
            for mname, mres in methods.items()
            if cname in mres["classifiers"]
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        order_str = "  >  ".join(f"{m} ({acc * 100:.2f}%)" for m, acc in ranked)
        lines.append(f"  {cname:<14}: {order_str}")
    lines.append("=" * 98)
    return "\n".join(lines)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_benchmark_plot(benchmark_results: Dict[str, Any], output_path: str) -> str:
    """Render a Gruvbox-dark grouped bar chart comparing Accuracy and Macro F1."""
    import matplotlib

    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    bg0_hard = "#1d2021"
    bg0 = "#282828"
    bg1 = "#3c3836"
    fg0 = "#fbf1c7"
    fg1 = "#ebdbb2"
    palette = ["#928374", "#83a598", "#fabd2f", "#b8bb26", "#d3869b", "#fe8019"]

    methods = list(benchmark_results["methods"].keys())
    classifiers: List[str] = []
    for mres in benchmark_results["methods"].values():
        for cname in mres["classifiers"]:
            if cname not in classifiers:
                classifiers.append(cname)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    fig.patch.set_facecolor(bg0_hard)
    x = np.arange(len(classifiers))
    width = min(0.80 / max(len(methods), 1), 0.24)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    multi_seed = len(benchmark_results["metadata"]["seeds"]) > 1

    panel_specs = [
        ("accuracy", "Top-1 Test Accuracy (%)"),
        ("macro_f1", "Macro F1 Score (%)"),
    ]
    for ax, (metric_key, ylabel) in zip(axes, panel_specs):
        ax.set_facecolor(bg0)
        ymin_vals: List[float] = []
        ymax_vals: List[float] = []
        for idx, mname in enumerate(methods):
            means = []
            stds = []
            for cname in classifiers:
                cnode = benchmark_results["methods"][mname]["classifiers"].get(cname)
                if cnode is None:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    s = cnode["summary"]
                    means.append(s[f"{metric_key}_mean"] * 100.0)
                    std_val = s[f"{metric_key}_std"] * 100.0
                    stds.append(std_val if (multi_seed and np.isfinite(std_val)) else 0.0)
            for mean_v, std_v in zip(means, stds):
                if np.isfinite(mean_v):
                    ymin_vals.append(mean_v - std_v)
                    ymax_vals.append(mean_v + std_v)
            bars = ax.bar(
                x + offsets[idx],
                means,
                width=width * 0.92,
                yerr=stds if multi_seed else None,
                capsize=3,
                color=palette[idx % len(palette)],
                edgecolor=bg0_hard,
                linewidth=0.8,
                error_kw={"ecolor": fg0, "elinewidth": 1.0},
                label=mname,
                zorder=3,
            )
            for bar, mean_v, std_v in zip(bars, means, stds):
                if np.isfinite(mean_v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        mean_v + std_v + (0.6 if not multi_seed else 1.0),
                        f"{mean_v:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=fg1,
                    )

        low = (
            max(0.0, float(np.floor((min(ymin_vals) - 6.0) / 5.0) * 5.0))
            if (ymin_vals and min(ymin_vals) > 50.0)
            else 0.0
        )
        top = max(105.0, (max(ymax_vals) + 5.0) if ymax_vals else 105.0)
        ax.set_ylim(low, top)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=10, color=fg0)
        ax.set_ylabel(ylabel, fontsize=10.5, color=fg0)
        ax.grid(axis="y", color=bg1, linestyle="--", linewidth=0.7, zorder=0)
        ax.tick_params(colors=fg1)
        for spine in ax.spines.values():
            spine.set_color(bg1)

    axes[0].set_title("Downstream Classification Accuracy", color=fg0, fontsize=11.5, pad=10)
    axes[1].set_title("Downstream Macro F1 (Class-Balanced)", color=fg0, fontsize=11.5, pad=10)
    axes[0].legend(
        facecolor=bg0,
        edgecolor=bg1,
        labelcolor=fg0,
        fontsize=8.5,
        loc="lower right",
    )

    dataset_name = benchmark_results["metadata"].get("dataset_name", "custom")
    fig.suptitle(
        f"Downstream Raman Classification Benchmark ({dataset_name})",
        color=fg0,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, facecolor=bg0_hard, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_benchmark_outputs(
    benchmark_results: Dict[str, Any],
    output_dir: str,
    prefix: str = "downstream_benchmark",
    save_plot: bool = True,
    save_predictions: bool = True,
) -> Dict[str, str]:
    """Write `.txt`, `.json`, `.csv`, optional `.png`, and optional `.npz` predictions."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    report_text = format_benchmark_report(benchmark_results)
    txt_path = os.path.join(output_dir, f"{prefix}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    paths["report_txt"] = txt_path

    # Strip large in-memory prediction arrays from JSON while keeping all metrics & CMs.
    json_payload = {
        "metadata": benchmark_results["metadata"],
        "class_names": benchmark_results["class_names"],
        "baseline_comparisons": benchmark_results.get("baseline_comparisons", {}),
        "methods": {},
    }
    npz_payload: Dict[str, np.ndarray] = {}
    if "y_test_encoded" in benchmark_results:
        npz_payload["y_test_encoded"] = np.asarray(
            benchmark_results["y_test_encoded"], dtype=np.int64
        )
    npz_payload["class_names"] = np.asarray(benchmark_results["class_names"])

    for mname, mres in benchmark_results["methods"].items():
        clean_classifiers: Dict[str, Any] = {}
        for cname, cdata in mres["classifiers"].items():
            clean_classifiers[cname] = {
                "summary": cdata["summary"],
                "per_seed": cdata["per_seed"],
            }
            for seed, preds in cdata.get("seed_predictions", {}).items():
                safe_key = f"{mname}__{cname}__seed{seed}".replace(" ", "_").replace("/", "_")
                npz_payload[safe_key] = np.asarray(preds, dtype=np.int64)
        json_payload["methods"][mname] = {
            "method": mres["method"],
            "input_dim": mres["input_dim"],
            "n_train": mres["n_train"],
            "n_val": mres["n_val"],
            "n_test": mres["n_test"],
            "n_classes": mres["n_classes"],
            "seeds": mres["seeds"],
            "preprocessing_time_sec": mres["preprocessing_time_sec"],
            "classifiers": clean_classifiers,
        }

    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(json_payload), f, indent=2)
    paths["results_json"] = json_path

    csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    comp_map = benchmark_results.get("baseline_comparisons", {}).get("comparisons", {})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "classifier",
                "input_dim",
                "n_seeds",
                "accuracy_mean",
                "accuracy_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "log_loss_mean",
                "log_loss_std",
                "roc_auc_ovr_mean",
                "fit_time_sec_mean",
                "prep_total_sec",
                "accuracy_delta_vs_baseline",
                "macro_f1_delta_vs_baseline",
                "mcnemar_p_median",
            ]
        )
        for mname, mres in benchmark_results["methods"].items():
            prep_t = mres["preprocessing_time_sec"]["total"]
            for cname, cdata in mres["classifiers"].items():
                s = cdata["summary"]
                comp = comp_map.get(mname, {}).get(cname, {})
                writer.writerow(
                    [
                        mname,
                        cname,
                        mres["input_dim"],
                        s["n_seeds"],
                        s["accuracy_mean"],
                        s["accuracy_std"],
                        s["balanced_accuracy_mean"],
                        s["balanced_accuracy_std"],
                        s["macro_f1_mean"],
                        s["macro_f1_std"],
                        s["log_loss_mean"],
                        s["log_loss_std"],
                        s["roc_auc_ovr_mean"],
                        s["fit_time_sec_mean"],
                        prep_t,
                        comp.get("accuracy_delta"),
                        comp.get("macro_f1_delta"),
                        comp.get("mcnemar_p_median"),
                    ]
                )
    paths["summary_csv"] = csv_path

    if save_predictions and npz_payload:
        npz_path = os.path.join(output_dir, f"{prefix}_predictions.npz")
        np.savez_compressed(npz_path, **npz_payload)
        paths["predictions_npz"] = npz_path

    if save_plot:
        plot_path = os.path.join(output_dir, f"{prefix}_plot.png")
        save_benchmark_plot(benchmark_results, plot_path)
        paths["plot_png"] = plot_path

    return paths


def run_downstream_benchmark(
    methods: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    val_fraction: float = 0.15,
    split_seed: int = 42,
    seeds: Sequence[int] = (42,),
    baseline_method: Optional[str] = None,
    dataset_name: str = "custom",
    classifier_factories: Optional[Mapping[str, Callable[[int], Any]]] = None,
    include_classifiers: Optional[Sequence[str]] = None,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 7,
    lr: float = 1e-3,
    device: Optional[str] = None,
    xgb_n_estimators: int = 500,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_early_stopping_rounds: Optional[int] = 30,
    xgb_device: str = "cpu",
    xgb_n_jobs: Optional[int] = None,
    linear_kwargs: Optional[Mapping[str, Any]] = None,
    xgb_kwargs: Optional[Mapping[str, Any]] = None,
    resnet_kwargs: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[str] = None,
    output_prefix: str = "downstream_benchmark",
    save_plot: bool = True,
    save_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Benchmark multiple Raman preprocessing methods under strictly identical controls.

    1. Encodes labels cleanly without test leakage.
    2. Carves a single deterministic stratified validation split if `X_val` is None
       and `val_fraction > 0` (only when `methods` aren't precomputed tuples/dicts).
    3. Runs `evaluate_method(...)` for every preprocessing method using identical
       splits, classifier architectures, fixed epoch budget, and random seeds.
    4. Computes paired deltas + exact McNemar significance vs `baseline_method`.
    5. Optionally saves `.txt`, `.json`, `.csv`, `.npz`, and `.png` outputs.
    """
    if not methods:
        raise ValueError("methods mapping must contain at least one preprocessing method.")

    X_tr = _validate_2d_spectra(X_train, "X_train")
    X_te = _validate_2d_spectra(X_test, "X_test")
    y_tr_enc, y_val_enc, y_te_enc, class_names = encode_labels(y_train, y_val, y_test)
    n_classes = len(class_names)

    has_precomputed = any(
        isinstance(p, (tuple, list, Mapping)) for p in methods.values()
    )
    if X_val is None:
        if has_precomputed:
            X_v = None
            y_v_enc = None
        elif val_fraction > 0.0:
            X_tr, X_v, y_tr_enc, y_v_enc = train_test_split(
                X_tr,
                y_tr_enc,
                test_size=val_fraction,
                stratify=y_tr_enc,
                random_state=split_seed,
            )
        else:
            X_v = None
            y_v_enc = None
    else:
        X_v = _validate_2d_spectra(X_val, "X_val")
        y_v_enc = y_val_enc

    resolved_device = resolve_device(device)
    method_results: Dict[str, Dict[str, Any]] = {}
    for mname, preprocessor in methods.items():
        method_results[mname] = evaluate_method(
            method_name=mname,
            X_train=X_tr,
            y_train=y_tr_enc,
            X_test=X_te,
            y_test=y_te_enc,
            X_val=X_v,
            y_val=y_v_enc,
            preprocessor=preprocessor,
            seeds=seeds,
            n_classes=n_classes,
            classifier_factories=classifier_factories,
            include_classifiers=include_classifiers,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            lr=lr,
            device=resolved_device,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
            xgb_early_stopping_rounds=xgb_early_stopping_rounds,
            xgb_device=xgb_device,
            xgb_n_jobs=xgb_n_jobs,
            linear_kwargs=linear_kwargs,
            xgb_kwargs=xgb_kwargs,
            resnet_kwargs=resnet_kwargs,
            return_models=False,
            verbose=verbose,
        )

    baseline_comp = add_baseline_comparisons(
        method_results, y_te_enc, baseline_method=baseline_method
    )
    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_v.shape[0]) if X_v is not None else 0,
            "n_test": int(X_te.shape[0]),
            "n_classes": n_classes,
            "seeds": [int(s) for s in seeds],
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "lr": lr,
            "device": resolved_device,
        },
        "class_names": class_names,
        "y_test_encoded": y_te_enc,
        "methods": method_results,
        "baseline_comparisons": baseline_comp,
    }

    report_text = format_benchmark_report(results)
    results["report_text"] = report_text
    if verbose:
        print("\n" + report_text, flush=True)

    if output_dir is not None:
        saved_paths = save_benchmark_outputs(
            results,
            output_dir=output_dir,
            prefix=output_prefix,
            save_plot=save_plot,
            save_predictions=save_predictions,
        )
        results["saved_paths"] = saved_paths
        if verbose:
            print(f"[Benchmark] Saved outputs to '{output_dir}': {saved_paths}", flush=True)

    return results
