"""
DOWNSTREAM CLASSIFICATION TESTING & EVALUATION SCRIPT - ROUND 12
----------------------------------------------------------------
Loads the 18 trained downstream models from `training_round12/downstream_models/`
and the cached preprocessed test splits from `training_round12/downstream_cache/`
(or regenerates them from `datasets/` if cache files are absent), runs forward-pass
inference on the held-out test sets, and writes comprehensive `.txt` reports,
`.csv`/`.json`/`.npz` artifacts, and Gruvbox Dark plots to `training_round12/downstream_results/`.

Designed to run either on OSCAR immediately after `train_downstream12.py` or locally
on macOS once `downstream_models/`, `downstream_logs/`, and `downstream_cache/` are synced.

Outputs Generated in `training_round12/downstream_results/`:
  - 18 per-run reports: `{dataset}_{classifier}_{prep}_test_results12.txt`
  - 2 per-dataset head-to-head reports:
      * `bacteria_id_downstream_comparison12.txt`
      * `mlrod_downstream_comparison12.txt`
  - 1 master summary report: `downstream_results_summary12.txt`
  - Structured data: `downstream_results12.csv`, `downstream_results12.json`, `downstream_predictions12.npz`
  - Plots: `downstream_accuracy_f1_comparison12.png`, `downstream_training_curves12.png`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Headless-safe for OSCAR compute nodes and local CLI runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

# Configure paths so imports resolve from project root or training_round12/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")
RAMAN_PLOTTING_SIBLING = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "raman plotting"))

for p in (PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR, RAMAN_PLOTTING_SIBLING):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from classifiers.benchmark import mcnemar_test
from classifiers.resnet1d import ResNet1D
from train_downstream12 import (
    CLASSIFIERS_ORDER,
    DATASETS_ORDER,
    PREPROCESSORS_ORDER,
    DualLogger,
    load_bacteria_id_dataset,
    load_mlrod_dataset,
    load_pgnet_v34_model,
    prepare_and_cache_preprocessed_splits,
    resolve_path,
    select_devices,
)

try:
    import xgboost as xgb
except Exception:
    xgb = None

GRUVBOX = {
    "bg0_hard": "#1d2021",
    "bg0": "#282828",
    "bg1": "#3c3836",
    "fg0": "#fbf1c7",
    "fg1": "#ebdbb2",
    "gray": "#928374",
    "red": "#fb4934",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "blue": "#83a598",
    "purple": "#d3869b",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
}

PREP_LABELS = {
    "raw": "Raw ([0,1] Norm)",
    "conventional": "Conventional (pre.py)",
    "pgnet_v34": "PolyGaussNet V3.4-8sec",
}

CLF_LABELS = {
    "linear": "Linear (LogReg)",
    "xgb": "XGBoost (Hist)",
    "resnet1d": "ResNet1D (Ho et al.)",
}


# =====================================================================
# 1. FORWARD-PASS INFERENCE & METRIC COMPUTATION
# =====================================================================
def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    if y_prob.shape[1] <= k:
        return 1.0
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    matches = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(matches))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    infer_sec: float,
) -> Dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    n_classes = len(class_names)
    labels_all = np.arange(n_classes)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0)
    )
    top3_acc = top_k_accuracy(y_true, y_prob, k=3)
    ce_loss = float(log_loss(y_true, np.clip(y_prob, 1e-12, 1.0), labels=labels_all))

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, zero_division=0
    )
    macro_prec = float(np.mean(prec_arr))
    macro_rec = float(np.mean(rec_arr))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    per_class = []
    for idx in range(n_classes):
        correct_i = int(cm[idx, idx])
        sup_i = int(sup_arr[idx])
        per_class.append(
            {
                "class_id": idx,
                "class_name": str(class_names[idx]),
                "precision": float(prec_arr[idx]),
                "recall": float(rec_arr[idx]),
                "f1": float(f1_arr[idx]),
                "support": sup_i,
                "correct": correct_i,
            }
        )

    n_samples = len(y_true)
    spec_per_s = float(n_samples / max(infer_sec, 1e-9))
    ms_per_spec = float((infer_sec * 1000.0) / max(n_samples, 1))

    return {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_acc,
        "log_loss": ce_loss,
        "infer_sec": float(infer_sec),
        "throughput_spec_per_s": spec_per_s,
        "ms_per_spectrum": ms_per_spec,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_linear_model(
    model_path: str, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    clf = joblib.load(model_path)
    t0 = time.perf_counter()
    y_prob = np.asarray(clf.predict_proba(X_test), dtype=np.float32)
    infer_sec = time.perf_counter() - t0
    metrics = compute_classification_metrics(y_test, y_prob, class_names, infer_sec)
    return {"primary": metrics, "best_val_ckpt": metrics, "last_epoch_ckpt": metrics}


def evaluate_xgb_model(
    model_path: str,
    meta: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    checkpoint_mode: str,
) -> Dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required to evaluate saved XGBoost models.")
    booster = xgb.Booster()
    booster.load_model(model_path)
    dtest = xgb.DMatrix(X_test)

    # Full matched rounds (`E_stop`) evaluation
    t0 = time.perf_counter()
    prob_last = np.asarray(booster.predict(dtest), dtype=np.float32)
    sec_last = time.perf_counter() - t0
    if prob_last.ndim == 1:
        prob_last = np.column_stack([1.0 - prob_last, prob_last])
    metrics_last = compute_classification_metrics(y_test, prob_last, class_names, sec_last)

    # Best validation accuracy round evaluation (within 1..E_stop)
    best_round = int(meta.get("best_epoch", meta.get("epochs_run", 1)))
    t1 = time.perf_counter()
    prob_best = np.asarray(
        booster.predict(dtest, iteration_range=(0, max(1, best_round))), dtype=np.float32
    )
    sec_best = time.perf_counter() - t1
    if prob_best.ndim == 1:
        prob_best = np.column_stack([1.0 - prob_best, prob_best])
    metrics_best = compute_classification_metrics(y_test, prob_best, class_names, sec_best)

    primary = metrics_best if checkpoint_mode == "best" else metrics_last
    return {
        "primary": primary,
        "best_val_ckpt": metrics_best,
        "last_epoch_ckpt": metrics_last,
    }


def _run_resnet1d_ckpt_inference(
    ckpt_path: str,
    X_test: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet1D(
        hidden_sizes=tuple(ckpt.get("hidden_sizes", [100] * 6)),
        num_blocks=tuple(ckpt.get("num_blocks", [2] * 6)),
        input_dim=int(ckpt["input_dim"]),
        in_channels=int(ckpt.get("in_channels", 64)),
        n_classes=int(ckpt["n_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_tensor = torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32)).unsqueeze(1)
    with torch.no_grad():
        warm = x_tensor[: min(4, len(x_tensor))].to(device)
        _ = model(warm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    probs_list: List[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            bx = x_tensor[i : i + batch_size].to(device, non_blocking=(device.type == "cuda"))
            logits = model(bx)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    return np.vstack(probs_list).astype(np.float32), elapsed


def evaluate_resnet1d_model(
    best_model_path: str,
    last_model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    device: torch.device,
    checkpoint_mode: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    prob_best, sec_best = _run_resnet1d_ckpt_inference(
        best_model_path, X_test, device=device, batch_size=batch_size
    )
    metrics_best = compute_classification_metrics(y_test, prob_best, class_names, sec_best)

    if os.path.exists(last_model_path):
        prob_last, sec_last = _run_resnet1d_ckpt_inference(
            last_model_path, X_test, device=device, batch_size=batch_size
        )
        metrics_last = compute_classification_metrics(y_test, prob_last, class_names, sec_last)
    else:
        metrics_last = metrics_best

    primary = metrics_best if checkpoint_mode == "best" else metrics_last
    return {
        "primary": primary,
        "best_val_ckpt": metrics_best,
        "last_epoch_ckpt": metrics_last,
    }


# =====================================================================
# 2. REPORT FORMATTING HELPERS
# =====================================================================
def sig_stars(p_val: float) -> str:
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def write_single_run_report(
    filepath: str,
    meta: Dict[str, Any],
    eval_bundle: Dict[str, Any],
    checkpoint_mode: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    primary = eval_bundle["primary"]
    best_m = eval_bundle["best_val_ckpt"]
    last_m = eval_bundle["last_epoch_ckpt"]

    with open(filepath, "w", encoding="utf-8") as f:
        w = 96
        f.write("=" * w + "\n")
        f.write(
            f" DOWNSTREAM TEST EVALUATION REPORT — {meta['run_id']}\n"
        )
        f.write("=" * w + "\n")
        f.write(f"  - Dataset:              {meta['dataset']}\n")
        f.write(f"  - Classifier:           {meta['classifier']} ({CLF_LABELS.get(meta['classifier'], '')})\n")
        f.write(f"  - Preprocessor:         {meta['preprocessor']} ({PREP_LABELS.get(meta['preprocessor'], '')})\n")
        f.write(f"  - Input Dimension:      {meta['input_dim']} bins | Classes: {meta['n_classes']}\n")
        f.write(f"  - Epochs / Rounds Run:  {meta['epochs_run']} (Best Val Epoch/Round: {meta['best_epoch']})\n")
        f.write(
            f"  - Validation Progress:  Best Val Acc = {meta['best_val_acc'] * 100:.2f}% "
            f"| Final Val Acc = {meta['final_val_acc'] * 100:.2f}% "
            f"| Final Val Loss = {meta['final_val_loss']:.5f}\n"
        )
        f.write(f"  - Training Wall Time:   {meta['train_time_sec']:.2f}s\n")
        f.write(f"  - Primary Eval Mode:    {checkpoint_mode.upper()}\n")
        f.write("=" * w + "\n\n")

        f.write("1. HELD-OUT TEST SET SUMMARY METRICS\n")
        f.write("-" * w + "\n")
        f.write(
            f"  {'Metric':28s} | {'Best-Val Checkpoint':>22s} | {'Final-Epoch (E_stop)':>22s}\n"
        )
        f.write("-" * w + "\n")
        rows = [
            ("Top-1 Accuracy (%)", f"{best_m['accuracy'] * 100:21.2f}%", f"{last_m['accuracy'] * 100:21.2f}%"),
            ("Balanced Accuracy (%)", f"{best_m['balanced_accuracy'] * 100:21.2f}%", f"{last_m['balanced_accuracy'] * 100:21.2f}%"),
            ("Macro F1-Score", f"{best_m['macro_f1']:22.4f}", f"{last_m['macro_f1']:22.4f}"),
            ("Weighted F1-Score", f"{best_m['weighted_f1']:22.4f}", f"{last_m['weighted_f1']:22.4f}"),
            ("Macro Precision", f"{best_m['macro_precision']:22.4f}", f"{last_m['macro_precision']:22.4f}"),
            ("Macro Recall", f"{best_m['macro_recall']:22.4f}", f"{last_m['macro_recall']:22.4f}"),
            ("Top-3 Accuracy (%)", f"{best_m['top3_accuracy'] * 100:21.2f}%", f"{last_m['top3_accuracy'] * 100:21.2f}%"),
            ("Test Cross-Entropy Loss", f"{best_m['log_loss']:22.5f}", f"{last_m['log_loss']:22.5f}"),
            ("Inference Time (s)", f"{best_m['infer_sec']:21.4f}s", f"{last_m['infer_sec']:21.4f}s"),
            ("Throughput (spectra/s)", f"{best_m['throughput_spec_per_s']:22.1f}", f"{last_m['throughput_spec_per_s']:22.1f}"),
        ]
        for label, v_best, v_last in rows:
            f.write(f"  {label:28s} | {v_best} | {v_last}\n")
        f.write("-" * w + "\n\n")

        f.write(f"2. PER-CLASS BREAKDOWN (Primary = {checkpoint_mode.upper()} Checkpoint, N = {primary['n_samples']})\n")
        f.write("-" * w + "\n")
        f.write(
            f"  {'ID':>3s} | {'Class Name':34s} | {'Precision':>9s} | {'Recall':>9s} | "
            f"{'F1-Score':>9s} | {'Correct / Support':>17s}\n"
        )
        f.write("-" * w + "\n")
        for pc in primary["per_class"]:
            f.write(
                f"  {pc['class_id']:3d} | {pc['class_name'][:34]:34s} | "
                f"{pc['precision']:9.4f} | {pc['recall']:9.4f} | {pc['f1']:9.4f} | "
                f"{pc['correct']:7d} / {pc['support']:<7d}\n"
            )
        f.write("-" * w + "\n\n")

        f.write("3. CONFUSION MATRIX (Rows = True Class ID, Columns = Predicted Class ID)\n")
        f.write("-" * w + "\n")
        cm = np.asarray(primary["confusion_matrix"], dtype=np.int64)
        n_c = cm.shape[0]
        header_cols = " ".join(f"{j:4d}" for j in range(n_c))
        f.write(f"  True\\Pred | {header_cols}\n")
        f.write("  " + "-" * (12 + 5 * n_c) + "\n")
        for i in range(n_c):
            row_str = " ".join(f"{cm[i, j]:4d}" for j in range(n_c))
            f.write(f"  {i:9d} | {row_str}\n")
        f.write("=" * w + "\n")


def write_dataset_comparison_report(
    filepath: str,
    ds_key: str,
    ds_info: Dict[str, Any],
    results_for_ds: Dict[Tuple[str, str], Dict[str, Any]],
    metas_for_ds: Dict[Tuple[str, str], Dict[str, Any]],
    y_test: np.ndarray,
    checkpoint_mode: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    w = 108
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * w + "\n")
        f.write(f" ROUND 12 DOWNSTREAM BENCHMARK COMPARISON — {ds_info.get('display_name', ds_key)}\n")
        f.write("=" * w + "\n")
        f.write(
            f"  - Dataset:        {ds_key} (train={ds_info.get('train_samples')}, "
            f"val={ds_info.get('val_samples')}, test={ds_info.get('test_samples')}, "
            f"bins={ds_info.get('wavenumber_bins')}, classes={ds_info.get('n_classes')})\n"
        )
        f.write("  - Normalization:  Per-spectrum [0, 1] min-max scaling applied upfront to all spectra\n")
        f.write(
            "  - Epoch Policy:   'raw' trained to convergence with patience; 'conventional' and 'pgnet_v34'\n"
            "                    trained for the exact same number of epochs/rounds within each classifier\n"
        )
        f.write("=" * w + "\n\n")

        # Preprocessing throughput summary if available
        prep_timings = ds_info.get("preprocessing_timings", {})
        if prep_timings:
            f.write("1. PREPROCESSING SPEED & THROUGHPUT\n")
            f.write("-" * w + "\n")
            f.write(
                f"  {'Preprocessor':26s} | {'Train (s)':>10s} | {'Val (s)':>10s} | "
                f"{'Test (s)':>10s} | {'Total (s)':>10s} | {'Throughput (spec/s)':>20s}\n"
            )
            f.write("-" * w + "\n")
            for prep in PREPROCESSORS_ORDER:
                pt = prep_timings.get(prep, {})
                f.write(
                    f"  {PREP_LABELS.get(prep, prep):26s} | "
                    f"{pt.get('train_sec', 0.0):10.2f} | "
                    f"{pt.get('val_sec', 0.0):10.2f} | "
                    f"{pt.get('test_sec', 0.0):10.2f} | "
                    f"{pt.get('total_sec', 0.0):10.2f} | "
                    f"{pt.get('throughput_spec_per_s', 0.0):20.1f}\n"
                )
            f.write("-" * w + "\n\n")

        # Table for BOTH Best-Val Checkpoint and Final-Epoch (`E_stop`) Checkpoint
        for sec_idx, (ckpt_key, ckpt_title) in enumerate(
            [
                ("best_val_ckpt", "2A. HELD-OUT TEST METRICS — BEST VALIDATION CHECKPOINT (WITHIN MATCHED EPOCH BUDGET)"),
                ("last_epoch_ckpt", "2B. HELD-OUT TEST METRICS — FINAL MATCHED EPOCH (E_stop) CHECKPOINT"),
            ]
        ):
            f.write(f"{ckpt_title}\n")
            f.write("-" * w + "\n")
            f.write(
                f"  {'Classifier':14s} | {'Preprocessor':22s} | {'Epochs':>6s} | {'BestEp':>6s} | "
                f"{'Test Acc':>9s} | {'Bal Acc':>9s} | {'Macro F1':>9s} | {'Weight F1':>9s} | "
                f"{'Top-3 Acc':>9s} | {'Test Loss':>9s}\n"
            )
            f.write("-" * w + "\n")
            for clf in CLASSIFIERS_ORDER:
                for prep in PREPROCESSORS_ORDER:
                    if (clf, prep) not in results_for_ds:
                        continue
                    m = results_for_ds[(clf, prep)][ckpt_key]
                    meta = metas_for_ds[(clf, prep)]
                    f.write(
                        f"  {clf:14s} | {prep:22s} | {meta['epochs_run']:6d} | {meta['best_epoch']:6d} | "
                        f"{m['accuracy'] * 100:8.2f}% | {m['balanced_accuracy'] * 100:8.2f}% | "
                        f"{m['macro_f1']:9.4f} | {m['weighted_f1']:9.4f} | "
                        f"{m['top3_accuracy'] * 100:8.2f}% | {m['log_loss']:9.5f}\n"
                    )
                f.write("-" * w + "\n")
            f.write("\n")

        # Pairwise Deltas & McNemar Significance Tests
        f.write(f"3. HEAD-TO-HEAD DELTAS & MCNEMAR SIGNIFICANCE TESTS (Mode = {checkpoint_mode.upper()})\n")
        f.write("-" * w + "\n")
        f.write(
            f"  {'Classifier':12s} | {'Comparison (A vs B)':28s} | {'dAcc (pp)':>10s} | "
            f"{'dMacroF1':>10s} | {'n10 (A>B)':>9s} | {'n01 (B>A)':>9s} | {'Discord':>8s} | {'p-value':>12s}\n"
        )
        f.write("-" * w + "\n")
        pairs = [
            ("pgnet_v34", "raw"),
            ("pgnet_v34", "conventional"),
            ("conventional", "raw"),
        ]
        for clf in CLASSIFIERS_ORDER:
            for prep_a, prep_b in pairs:
                if (clf, prep_a) not in results_for_ds or (clf, prep_b) not in results_for_ds:
                    continue
                ma = results_for_ds[(clf, prep_a)]["primary"]
                mb = results_for_ds[(clf, prep_b)]["primary"]
                d_acc = (ma["accuracy"] - mb["accuracy"]) * 100.0
                d_f1 = ma["macro_f1"] - mb["macro_f1"]
                mc = mcnemar_test(y_test, ma["y_pred"], mb["y_pred"])
                stars = sig_stars(mc["p_value"])
                f.write(
                    f"  {clf:12s} | {f'{prep_a} vs {prep_b}':28s} | {d_acc:+9.2f}% | "
                    f"{d_f1:+10.4f} | {mc['method_only_correct']:9d} | {mc['baseline_only_correct']:9d} | "
                    f"{mc['discordant_pairs']:8d} | {mc['p_value']:8.2e} ({stars})\n"
                )
            f.write("-" * w + "\n")
        f.write("=" * w + "\n")


# =====================================================================
# 3. PLOTTING (GRUVBOX DARK)
# =====================================================================
def plot_downstream_comparison(
    png_path: str,
    summary_rows: List[Dict[str, Any]],
) -> None:
    if not summary_rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in summary_rows), key=lambda x: DATASETS_ORDER.index(x) if x in DATASETS_ORDER else 99)
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 2, figsize=(15, 5.5 * n_ds), dpi=180, squeeze=False)
    fig.patch.set_facecolor(GRUVBOX["bg0_hard"])

    prep_colors = {
        "raw": GRUVBOX["gray"],
        "conventional": GRUVBOX["blue"],
        "pgnet_v34": GRUVBOX["orange"],
    }

    for row_idx, ds in enumerate(datasets):
        ds_rows = {
            (r["classifier"], r["preprocessor"]): r
            for r in summary_rows
            if r["dataset"] == ds
        }
        clfs = [c for c in CLASSIFIERS_ORDER if any((c, p) in ds_rows for p in PREPROCESSORS_ORDER)]
        x = np.arange(len(clfs))
        width = 0.25

        for col_idx, (metric_key, metric_label, scale) in enumerate(
            [
                ("test_accuracy", "Test Top-1 Accuracy (%)", 100.0),
                ("test_macro_f1", "Test Macro F1-Score (%)", 100.0),
            ]
        ):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(GRUVBOX["bg0"])
            for spine in ax.spines.values():
                spine.set_color(GRUVBOX["bg1"])
            ax.tick_params(colors=GRUVBOX["fg1"], labelsize=10)
            ax.grid(axis="y", color=GRUVBOX["bg1"], linestyle="--", alpha=0.6, zorder=0)

            all_vals = []
            for p_idx, prep in enumerate(PREPROCESSORS_ORDER):
                vals = [
                    ds_rows[(c, prep)][metric_key] * scale if (c, prep) in ds_rows else 0.0
                    for c in clfs
                ]
                all_vals.extend(vals)
                offset = (p_idx - 1) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width=width * 0.92,
                    label=PREP_LABELS.get(prep, prep),
                    color=prep_colors.get(prep, GRUVBOX["aqua"]),
                    edgecolor=GRUVBOX["fg0"],
                    linewidth=0.7,
                    zorder=3,
                )
                for b in bars:
                    h = b.get_height()
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        h + 0.4,
                        f"{h:.1f}",
                        ha="center",
                        va="bottom",
                        color=GRUVBOX["fg0"],
                        fontsize=8,
                        fontweight="bold",
                    )

            y_min = max(0.0, min(all_vals) - 8.0) if all_vals else 0.0
            ax.set_ylim(y_min, 104.5)
            ax.set_xticks(x)
            ax.set_xticklabels([CLF_LABELS.get(c, c) for c in clfs], color=GRUVBOX["fg0"], fontweight="bold")
            ax.set_ylabel(metric_label, color=GRUVBOX["fg1"], fontsize=10)
            ax.set_title(
                f"{ds.upper()} — {metric_label}",
                color=GRUVBOX["yellow"],
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax.legend(
                facecolor=GRUVBOX["bg1"],
                edgecolor=GRUVBOX["gray"],
                labelcolor=GRUVBOX["fg0"],
                fontsize=8.5,
                loc="lower right",
            )

    fig.suptitle(
        "Round 12 Downstream Classification Benchmark (Raw vs Conventional vs PolyGaussNet V3.4-8sec)",
        color=GRUVBOX["fg0"],
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_training_curves(
    png_path: str,
    log_dir: str,
    datasets: List[str],
) -> None:
    iter_clfs = ["xgb", "resnet1d"]
    n_rows = len(datasets) * len(iter_clfs)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.0 * n_rows), dpi=180, squeeze=False)
    fig.patch.set_facecolor(GRUVBOX["bg0_hard"])

    prep_colors = {
        "raw": GRUVBOX["gray"],
        "conventional": GRUVBOX["blue"],
        "pgnet_v34": GRUVBOX["orange"],
    }

    row_cursor = 0
    plotted_any = False
    for ds in datasets:
        for clf in iter_clfs:
            ax_loss = axes[row_cursor, 0]
            ax_acc = axes[row_cursor, 1]
            for ax in (ax_loss, ax_acc):
                ax.set_facecolor(GRUVBOX["bg0"])
                for spine in ax.spines.values():
                    spine.set_color(GRUVBOX["bg1"])
                ax.tick_params(colors=GRUVBOX["fg1"], labelsize=9)
                ax.grid(True, color=GRUVBOX["bg1"], linestyle="--", alpha=0.6)

            for prep in PREPROCESSORS_ORDER:
                csv_file = os.path.join(log_dir, f"{ds}_{clf}_{prep}_training_history12.csv")
                if not os.path.exists(csv_file):
                    continue
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                if not reader:
                    continue
                plotted_any = True
                epochs = [int(r["epoch"]) for r in reader]
                tr_loss = [float(r["train_loss"]) for r in reader]
                va_loss = [float(r["val_loss"]) for r in reader]
                va_acc = [float(r["val_acc"]) * 100.0 for r in reader]
                color = prep_colors.get(prep, GRUVBOX["aqua"])
                label_base = PREP_LABELS.get(prep, prep)

                ax_loss.plot(epochs, tr_loss, linestyle="--", alpha=0.55, color=color, linewidth=1.3)
                ax_loss.plot(epochs, va_loss, linestyle="-", color=color, linewidth=2.0, label=f"{label_base} (Val)")
                ax_acc.plot(epochs, va_acc, linestyle="-", color=color, linewidth=2.0, label=f"{label_base} (Val)")

            x_lbl = "Boosting Round" if clf == "xgb" else "Epoch"
            ax_loss.set_title(
                f"{ds.upper()} | {CLF_LABELS.get(clf, clf)} — Loss Progression (Dashed=Train, Solid=Val)",
                color=GRUVBOX["yellow"],
                fontsize=11,
                fontweight="bold",
            )
            ax_loss.set_xlabel(x_lbl, color=GRUVBOX["fg1"])
            ax_loss.set_ylabel("Cross-Entropy Loss", color=GRUVBOX["fg1"])
            ax_loss.legend(facecolor=GRUVBOX["bg1"], edgecolor=GRUVBOX["gray"], labelcolor=GRUVBOX["fg0"], fontsize=8)

            ax_acc.set_title(
                f"{ds.upper()} | {CLF_LABELS.get(clf, clf)} — Validation Accuracy (%)",
                color=GRUVBOX["yellow"],
                fontsize=11,
                fontweight="bold",
            )
            ax_acc.set_xlabel(x_lbl, color=GRUVBOX["fg1"])
            ax_acc.set_ylabel("Validation Accuracy (%)", color=GRUVBOX["fg1"])
            ax_acc.legend(facecolor=GRUVBOX["bg1"], edgecolor=GRUVBOX["gray"], labelcolor=GRUVBOX["fg0"], fontsize=8)

            row_cursor += 1

    if plotted_any:
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        fig.suptitle(
            "Round 12 Downstream Training & Validation Progression (Matched Epoch/Round Budget)",
            color=GRUVBOX["fg0"],
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(png_path, facecolor=fig.get_facecolor())
    plt.close(fig)


# =====================================================================
# 4. MAIN EVALUATION DRIVER
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 12 Downstream Classification Testing & Evaluation Suite (Program 2)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="training_round12",
        help="Root directory containing downstream_models/, downstream_logs/, downstream_cache/.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS_ORDER),
        default=list(DATASETS_ORDER),
        help="Datasets to evaluate (default: bacteria_id mlrod).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIERS_ORDER),
        default=list(CLASSIFIERS_ORDER),
        help="Classifiers to evaluate (default: linear xgb resnet1d).",
    )
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Primary checkpoint for ResNet1D/XGBoost ('best' val acc or 'last' matched epoch E_stop; both are reported in tables).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device for ResNet1D / PGNet fallback inference ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--bacteria-dir",
        type=str,
        default=os.path.join("datasets", "bacteria-ID"),
        help="Fallback path to datasets/bacteria-ID if cache is missing.",
    )
    parser.add_argument(
        "--mlrod-dir",
        type=str,
        default=os.path.join("datasets", "MLROD"),
        help="Fallback path to datasets/MLROD if cache is missing.",
    )
    parser.add_argument(
        "--pgnet-model",
        type=str,
        default=os.path.join("training_round12", "polygaussnet12.pth"),
        help="Fallback path to PolyGaussNet V3.4 weights if cache is missing.",
    )
    parser.add_argument(
        "--pgnet-ckpt",
        type=str,
        default=os.path.join("training_round12", "training_checkpoint12.pth"),
        help="Fallback checkpoint for PolyGaussNet V3.4.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="CPU workers if fallback conventional preprocessing is needed.",
    )
    parser.add_argument(
        "--require-all-runs",
        action="store_true",
        help="Safely defer evaluation (exit 0) if not all requested runs have completed training (e.g. Slurm preemption).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating .png plots.",
    )
    return parser.parse_args()


def load_test_splits_for_dataset(
    ds_key: str,
    cache_dir: str,
    args: argparse.Namespace,
    pt_device: torch.device,
    master_log: DualLogger,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Load preprocessed test splits from downstream_cache/, or regenerate if missing."""
    y_test_path = os.path.join(cache_dir, f"{ds_key}_y_test.npy")
    cls_path = os.path.join(cache_dir, f"{ds_key}_class_names.npy")
    prep_paths = {
        prep: os.path.join(cache_dir, f"{ds_key}_test_{prep}.npy")
        for prep in PREPROCESSORS_ORDER
    }

    if (
        os.path.exists(y_test_path)
        and os.path.exists(cls_path)
        and all(os.path.exists(p) for p in prep_paths.values())
    ):
        y_test = np.load(y_test_path).astype(np.int64)
        class_names = [str(x) for x in np.load(cls_path, allow_pickle=True).tolist()]
        X_test_by_prep = {
            prep: np.load(p).astype(np.float32, copy=False)
            for prep, p in prep_paths.items()
        }
        master_log.log(
            f"  [Cache Loaded] {ds_key}: {len(y_test)} test spectra x {X_test_by_prep['raw'].shape[1]} bins "
            f"({len(class_names)} classes)"
        )
        return X_test_by_prep, y_test, class_names

    master_log.log(
        f"  [Cache Miss] Regenerating preprocessed test splits for '{ds_key}' from raw dataset..."
    )
    if ds_key == "bacteria_id":
        dataset = load_bacteria_id_dataset(args.bacteria_dir)
    elif ds_key == "mlrod":
        dataset = load_mlrod_dataset(args.mlrod_dir, subset_mode="pure12")
    else:
        raise ValueError(f"Unsupported dataset: {ds_key}")

    pgnet_model, _, _, _ = load_pgnet_v34_model(args.pgnet_model, args.pgnet_ckpt, device=pt_device)
    splits_by_prep, _ = prepare_and_cache_preprocessed_splits(
        dataset=dataset,
        cache_dir=cache_dir,
        pgnet_model=pgnet_model,
        device=pt_device,
        pgnet_batch_size=512,
        n_jobs=args.n_jobs,
        cache_train=False,
        use_existing_cache=True,
        master_log=master_log,
    )
    X_test_by_prep = {prep: splits_by_prep[prep]["test"] for prep in PREPROCESSORS_ORDER}
    return X_test_by_prep, dataset["y_test"], dataset["class_names"]


def main() -> int:
    args = parse_args()
    out_root = resolve_path(args.output_root)
    log_dir = os.path.join(out_root, "downstream_logs")
    model_dir = os.path.join(out_root, "downstream_models")
    cache_dir = os.path.join(out_root, "downstream_cache")
    results_dir = os.path.join(out_root, "downstream_results")
    os.makedirs(results_dir, exist_ok=True)

    summary_txt_path = os.path.join(results_dir, "downstream_results_summary12.txt")
    summary_csv_path = os.path.join(results_dir, "downstream_results12.csv")
    summary_json_path = os.path.join(results_dir, "downstream_results12.json")
    preds_npz_path = os.path.join(results_dir, "downstream_predictions12.npz")
    plot_comp_path = os.path.join(results_dir, "downstream_accuracy_f1_comparison12.png")
    plot_curves_path = os.path.join(results_dir, "downstream_training_curves12.png")

    if args.require_all_runs:
        missing_runs = [
            f"{ds}_{clf}_{prep}"
            for ds in args.datasets
            for clf in args.classifiers
            for prep in PREPROCESSORS_ORDER
            if not os.path.exists(os.path.join(model_dir, f"{ds}_{clf}_{prep}_meta.json"))
        ]
        if missing_runs:
            print(
                f"[DEFERRED] {len(missing_runs)} run(s) not yet finished ({missing_runs[:4]}...). "
                "Resubmit training job to finish remaining runs before evaluation.",
                flush=True,
            )
            return 0

    master_log = DualLogger(summary_txt_path)
    try:
        pt_device, _ = select_devices(args.device, "cpu")
        manifest_path = os.path.join(log_dir, "training_manifest12.json")
        manifest: Dict[str, Any] = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        w = 108
        master_log.log("=" * w)
        master_log.log(" ROUND 12 DOWNSTREAM CLASSIFICATION TEST EVALUATION SUITE")
        master_log.log("=" * w)
        master_log.log(f"  - Evaluation Device:    {pt_device}")
        master_log.log(f"  - Primary Checkpoint:   {args.checkpoint_mode.upper()} (tables report both BEST_VAL and FINAL_EPOCH)")
        master_log.log(f"  - Models Directory:     {model_dir}")
        master_log.log(f"  - Cache Directory:      {cache_dir}")
        master_log.log(f"  - Results Directory:    {results_dir}")
        master_log.log("=" * w)

        summary_rows: List[Dict[str, Any]] = []
        npz_arrays: Dict[str, np.ndarray] = {}
        full_json_payload: Dict[str, Any] = {
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_mode": args.checkpoint_mode,
            "datasets": {},
            "runs": {},
            "mcnemar_tests": {},
        }

        for ds_key in args.datasets:
            master_log.log(f"\n>>> Evaluating Dataset: {ds_key}")
            X_test_by_prep, y_test, class_names = load_test_splits_for_dataset(
                ds_key, cache_dir, args, pt_device, master_log
            )
            npz_arrays[f"{ds_key}_y_test"] = y_test

            ds_info = manifest.get("datasets", {}).get(
                ds_key,
                {
                    "display_name": ds_key,
                    "n_classes": len(class_names),
                    "wavenumber_bins": int(X_test_by_prep["raw"].shape[1]),
                    "test_samples": int(len(y_test)),
                },
            )

            results_for_ds: Dict[Tuple[str, str], Dict[str, Any]] = {}
            metas_for_ds: Dict[Tuple[str, str], Dict[str, Any]] = {}

            for clf_key in args.classifiers:
                for prep_key in PREPROCESSORS_ORDER:
                    run_id = f"{ds_key}_{clf_key}_{prep_key}"
                    meta_path = os.path.join(model_dir, f"{run_id}_meta.json")
                    if not os.path.exists(meta_path):
                        master_log.log(f"  [SKIP] Metadata not found for '{run_id}' at {meta_path}")
                        continue

                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    X_te = X_test_by_prep[prep_key]

                    if clf_key == "linear":
                        model_file = os.path.join(model_dir, f"{run_id}.joblib")
                        eval_bundle = evaluate_linear_model(model_file, X_te, y_test, class_names)
                    elif clf_key == "xgb":
                        model_file = os.path.join(model_dir, f"{run_id}.json")
                        eval_bundle = evaluate_xgb_model(
                            model_file, meta, X_te, y_test, class_names, args.checkpoint_mode
                        )
                    elif clf_key == "resnet1d":
                        best_file = os.path.join(model_dir, f"{run_id}_best.pth")
                        last_file = os.path.join(model_dir, f"{run_id}_last.pth")
                        eval_bundle = evaluate_resnet1d_model(
                            best_file,
                            last_file,
                            X_te,
                            y_test,
                            class_names,
                            device=pt_device,
                            checkpoint_mode=args.checkpoint_mode,
                        )
                    else:
                        raise ValueError(f"Unknown classifier: {clf_key}")

                    results_for_ds[(clf_key, prep_key)] = eval_bundle
                    metas_for_ds[(clf_key, prep_key)] = meta

                    # Save per-run detailed .txt report
                    run_report_path = os.path.join(results_dir, f"{run_id}_test_results12.txt")
                    write_single_run_report(run_report_path, meta, eval_bundle, args.checkpoint_mode)

                    prim = eval_bundle["primary"]
                    best_m = eval_bundle["best_val_ckpt"]
                    last_m = eval_bundle["last_epoch_ckpt"]

                    npz_arrays[f"{run_id}_y_pred_best"] = best_m["y_pred"]
                    npz_arrays[f"{run_id}_y_pred_last"] = last_m["y_pred"]
                    npz_arrays[f"{run_id}_y_prob_primary"] = prim["y_prob"]

                    master_log.log(
                        f"  [{run_id:34s}] Epochs={meta['epochs_run']:3d} (BestEp={meta['best_epoch']:3d}) | "
                        f"Test Acc (BestCkpt)={best_m['accuracy'] * 100:6.2f}% | "
                        f"Test Acc (FinalEp)={last_m['accuracy'] * 100:6.2f}% | "
                        f"Macro F1={prim['macro_f1']:.4f}"
                    )

                    row_dict = {
                        "run_id": run_id,
                        "dataset": ds_key,
                        "classifier": clf_key,
                        "preprocessor": prep_key,
                        "epochs_run": int(meta["epochs_run"]),
                        "best_epoch": int(meta["best_epoch"]),
                        "best_val_acc": float(meta["best_val_acc"]),
                        "final_val_acc": float(meta["final_val_acc"]),
                        "test_accuracy": float(prim["accuracy"]),
                        "test_balanced_accuracy": float(prim["balanced_accuracy"]),
                        "test_macro_f1": float(prim["macro_f1"]),
                        "test_weighted_f1": float(prim["weighted_f1"]),
                        "test_top3_accuracy": float(prim["top3_accuracy"]),
                        "test_log_loss": float(prim["log_loss"]),
                        "best_ckpt_test_accuracy": float(best_m["accuracy"]),
                        "best_ckpt_test_macro_f1": float(best_m["macro_f1"]),
                        "last_ckpt_test_accuracy": float(last_m["accuracy"]),
                        "last_ckpt_test_macro_f1": float(last_m["macro_f1"]),
                        "infer_sec": float(prim["infer_sec"]),
                        "throughput_spec_per_s": float(prim["throughput_spec_per_s"]),
                        "train_time_sec": float(meta["train_time_sec"]),
                    }
                    summary_rows.append(row_dict)

                    # Store JSON-serializable summary (without raw numpy arrays)
                    full_json_payload["runs"][run_id] = {
                        **row_dict,
                        "per_class": prim["per_class"],
                        "confusion_matrix": prim["confusion_matrix"],
                    }

            # Save per-dataset head-to-head comparison .txt report
            ds_comp_path = os.path.join(results_dir, f"{ds_key}_downstream_comparison12.txt")
            write_dataset_comparison_report(
                ds_comp_path,
                ds_key,
                ds_info,
                results_for_ds,
                metas_for_ds,
                y_test,
                args.checkpoint_mode,
            )
            with open(ds_comp_path, "r", encoding="utf-8") as f_comp:
                master_log.log("\n" + f_comp.read())

            # Record McNemar tests in JSON payload
            full_json_payload["mcnemar_tests"][ds_key] = {}
            for clf_key in args.classifiers:
                full_json_payload["mcnemar_tests"][ds_key][clf_key] = {}
                for pa, pb in [("pgnet_v34", "raw"), ("pgnet_v34", "conventional"), ("conventional", "raw")]:
                    if (clf_key, pa) in results_for_ds and (clf_key, pb) in results_for_ds:
                        mc = mcnemar_test(
                            y_test,
                            results_for_ds[(clf_key, pa)]["primary"]["y_pred"],
                            results_for_ds[(clf_key, pb)]["primary"]["y_pred"],
                        )
                        full_json_payload["mcnemar_tests"][ds_key][clf_key][f"{pa}_vs_{pb}"] = mc

        # Save CSV, JSON, and NPZ artifacts
        if summary_rows:
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(full_json_payload, f, indent=2)

        np.savez_compressed(preds_npz_path, **npz_arrays)

        if not args.no_plots and summary_rows:
            plot_downstream_comparison(plot_comp_path, summary_rows)
            plot_training_curves(plot_curves_path, log_dir, args.datasets)

        master_log.log("\n" + "=" * w)
        master_log.log(" SAVED DOWNSTREAM EVALUATION ARTIFACTS")
        master_log.log("=" * w)
        master_log.log(f"  - Master Summary Report:   {summary_txt_path}")
        master_log.log(f"  - Summary CSV Table:       {summary_csv_path}")
        master_log.log(f"  - Summary JSON Metrics:    {summary_json_path}")
        master_log.log(f"  - Predictions Archive:     {preds_npz_path}")
        if not args.no_plots:
            master_log.log(f"  - Comparison Bar Plot:     {plot_comp_path}")
            master_log.log(f"  - Training Curves Plot:    {plot_curves_path}")
        master_log.log("=" * w)
        return 0
    finally:
        master_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
