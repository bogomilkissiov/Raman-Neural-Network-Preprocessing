"""Shared downstream classifier & benchmarking suite for Raman preprocessing."""

from .benchmark import (
    add_baseline_comparisons,
    compute_classification_metrics,
    default_n_jobs,
    encode_labels,
    evaluate_method,
    format_benchmark_report,
    make_default_classifiers,
    mcnemar_test,
    resolve_device,
    run_downstream_benchmark,
    save_benchmark_outputs,
    save_benchmark_plot,
    stratified_train_val_test_split,
)
from .linear import LinearSpectralClassifier
from .resnet1d import BacteriaResNetClassifier, ResidualBlock, ResNet1D
from .xgb import XGBSpectralClassifier

__all__ = [
    "LinearSpectralClassifier",
    "XGBSpectralClassifier",
    "ResidualBlock",
    "ResNet1D",
    "BacteriaResNetClassifier",
    "evaluate_method",
    "run_downstream_benchmark",
    "make_default_classifiers",
    "stratified_train_val_test_split",
    "encode_labels",
    "compute_classification_metrics",
    "mcnemar_test",
    "add_baseline_comparisons",
    "format_benchmark_report",
    "save_benchmark_plot",
    "save_benchmark_outputs",
    "default_n_jobs",
    "resolve_device",
]
