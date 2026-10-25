"""
HEAD-TO-HEAD BENCHMARK SCRIPT - ROUND 7 (PolyGaussNet V3 vs PolyGaussNet V3.1)
-----------------------------------------------------------------------------
Evaluates PolyGaussNet3 vs PolyGaussNet3.1 on held-out testing_data7:
  - 5 Bin Sizes: 800, 1016, 1200, 1400, 1600 bins (1,024 spectra each = 5,120 total)
  - Dual Evaluation Modes:
    1. RAW VALIDATION: Direct output vs ground truth pure spectra.
    2. BASELINE-SHIFTED VALIDATION: Exact identical DC mean alignment for both models.
  - Generates comprehensive validation report: validation_results_head_to_head.txt
-----------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# Resolve imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1

DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "testing_data7")
DEFAULT_V3_PATH = os.path.join(SCRIPT_DIR, "polygaussnet3.pth")
DEFAULT_V3_1_PATH = os.path.join(SCRIPT_DIR, "polygaussnet3_1.pth")
DEFAULT_REPORT_PATH = os.path.join(SCRIPT_DIR, "validation_results_head_to_head.txt")
BIN_SIZES = [800, 1016, 1200, 1400, 1600]
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31


def load_test_chunk(filepath: str):
    with np.load(filepath) as data:
        raw = data["full_matrix"].astype(np.float32)
        pure = data["pure_matrix"].astype(np.float32)
        bc_target = data["pure_noise_cosmic_matrix"].astype(np.float32)
        if "baseline_matrix" in data:
            baseline = data["baseline_matrix"].astype(np.float32)
        else:
            baseline = raw - bc_target
    return raw, pure, baseline, bc_target


def align_shift(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean baseline alignment shift to isolate pure spectral shape."""
    offset = np.mean(target, axis=-1, keepdims=True) - np.mean(pred, axis=-1, keepdims=True)
    return pred + offset


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    
    # Cosine Similarity
    dot = np.sum(pred * target, axis=-1)
    norm_p = np.linalg.norm(pred, axis=-1) + 1e-8
    norm_t = np.linalg.norm(target, axis=-1) + 1e-8
    cos_sim = float(np.mean(dot / (norm_p * norm_t)))
    
    return mse, mae, cos_sim


def evaluate_model_on_spectra(model, raw_np: np.ndarray, device: torch.device, batch_size: int = 256):
    model.eval()
    n_samples = raw_np.shape[0]
    clean_preds = []
    base_preds = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(raw_np[i : i + batch_size]).unsqueeze(1).to(device)
            clean, base, _, _ = model(batch)
            clean_preds.append(clean.squeeze(1).cpu().numpy())
            base_preds.append(base.squeeze(1).cpu().numpy())
            
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(clean_preds, axis=0), np.concatenate(base_preds, axis=0)


def run_benchmark():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Compute Device: {device}")

    # Load Model V3
    print(f"\nLoading PolyGaussNet V3 from {DEFAULT_V3_PATH}...")
    model_v3 = PolyGaussNet3(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
    model_v3.load_state_dict(torch.load(DEFAULT_V3_PATH, map_location=device))
    params_v3 = sum(p.numel() for p in model_v3.parameters())

    # Load Model V3.1
    print(f"Loading PolyGaussNet V3.1 from {DEFAULT_V3_1_PATH}...")
    model_v3_1 = PolyGaussNet3_1(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
    model_v3_1.load_state_dict(torch.load(DEFAULT_V3_1_PATH, map_location=device))
    params_v3_1 = sum(p.numel() for p in model_v3_1.parameters())

    print(f"  V3.0 Parameters: {params_v3:,}")
    print(f"  V3.1 Parameters: {params_v3_1:,} ({((params_v3_1 - params_v3)/params_v3)*100:+.1f}%)")

    # Storage for per-bin metrics
    results = {}

    total_samples = 0
    all_raw_mse_v3, all_raw_mse_v3_1 = [], []
    all_shf_mse_v3, all_shf_mse_v3_1 = [], []
    all_base_mse_v3, all_base_mse_v3_1 = [], []
    all_cos_v3, all_cos_v3_1 = [], []

    for b in BIN_SIZES:
        bin_dir = os.path.join(DEFAULT_DATA_DIR, str(b))
        chunk_files = sorted(glob.glob(os.path.join(bin_dir, "*.npz")))
        if not chunk_files:
            print(f"Warning: No test chunk found for {b} bins in {bin_dir}!")
            continue
        
        raw, pure, base_true, _ = load_test_chunk(chunk_files[0])
        n = raw.shape[0]
        total_samples += n

        # Run inference
        clean_v3, base_v3 = evaluate_model_on_spectra(model_v3, raw, device)
        clean_v3_1, base_v3_1 = evaluate_model_on_spectra(model_v3_1, raw, device)

        # Baseline shifted predictions
        clean_v3_shf = align_shift(clean_v3, pure)
        clean_v3_1_shf = align_shift(clean_v3_1, pure)

        # Metrics V3
        raw_mse_v3, raw_mae_v3, cos_v3 = compute_metrics(clean_v3, pure)
        shf_mse_v3, shf_mae_v3, _ = compute_metrics(clean_v3_shf, pure)
        base_mse_v3, _, _ = compute_metrics(base_v3, base_true)

        # Metrics V3.1
        raw_mse_v3_1, raw_mae_v3_1, cos_v3_1 = compute_metrics(clean_v3_1, pure)
        shf_mse_v3_1, shf_mae_v3_1, _ = compute_metrics(clean_v3_1_shf, pure)
        base_mse_v3_1, _, _ = compute_metrics(base_v3_1, base_true)

        results[b] = {
            "n": n,
            "v3": {"raw_mse": raw_mse_v3, "raw_mae": raw_mae_v3, "shf_mse": shf_mse_v3, "shf_mae": shf_mae_v3, "base_mse": base_mse_v3, "cos": cos_v3},
            "v3_1": {"raw_mse": raw_mse_v3_1, "raw_mae": raw_mae_v3_1, "shf_mse": shf_mse_v3_1, "shf_mae": shf_mae_v3_1, "base_mse": base_mse_v3_1, "cos": cos_v3_1},
        }

        all_raw_mse_v3.append(raw_mse_v3)
        all_raw_mse_v3_1.append(raw_mse_v3_1)
        all_shf_mse_v3.append(shf_mse_v3)
        all_shf_mse_v3_1.append(shf_mse_v3_1)
        all_base_mse_v3.append(base_mse_v3)
        all_base_mse_v3_1.append(base_mse_v3_1)
        all_cos_v3.append(cos_v3)
        all_cos_v3_1.append(cos_v3_1)

    # Calculate overall means
    mean_raw_mse_v3 = np.mean(all_raw_mse_v3)
    mean_raw_mse_v3_1 = np.mean(all_raw_mse_v3_1)
    mean_shf_mse_v3 = np.mean(all_shf_mse_v3)
    mean_shf_mse_v3_1 = np.mean(all_shf_mse_v3_1)
    mean_base_mse_v3 = np.mean(all_base_mse_v3)
    mean_base_mse_v3_1 = np.mean(all_base_mse_v3_1)
    mean_cos_v3 = np.mean(all_cos_v3)
    mean_cos_v3_1 = np.mean(all_cos_v3_1)

    diff_raw_mse = ((mean_raw_mse_v3_1 - mean_raw_mse_v3) / mean_raw_mse_v3) * 100
    diff_shf_mse = ((mean_shf_mse_v3_1 - mean_shf_mse_v3) / mean_shf_mse_v3) * 100
    diff_base_mse = ((mean_base_mse_v3_1 - mean_base_mse_v3) / mean_base_mse_v3) * 100

    # Build report string
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(" ROUND 7: HEAD-TO-HEAD VALIDATION REPORT (PolyGaussNet V3 vs PolyGaussNet V3.1) ")
    report_lines.append("=" * 100)
    report_lines.append(f"Total Test Spectra: {total_samples:,} ({len(BIN_SIZES)} resolutions x 1,024 spectra)")
    report_lines.append(f"PolyGaussNet V3.0 Parameters:   {params_v3:,}")
    report_lines.append(f"PolyGaussNet V3.1 Parameters:   {params_v3_1:,} ({(params_v3_1-params_v3)/params_v3*100:+.1f}%)")
    report_lines.append(f"Padding Mode (Both Models):     Zero-Padding (Strict Controlled Architectural Experiment)")
    report_lines.append(f"Polynomial Degree (Both Models): Degree {POLY_ORDER}")
    report_lines.append("-" * 100)
    report_lines.append("\nOVERALL BENCHMARK SUMMARY:")
    report_lines.append(f"{'Metric':<30} | {'PolyGaussNet V3.0':<20} | {'PolyGaussNet V3.1':<20} | {'Difference (V3.1 vs V3)':<22}")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Raw Validation Pure MSE':<30} | {mean_raw_mse_v3:<20.6f} | {mean_raw_mse_v3_1:<20.6f} | {diff_raw_mse:+.2f}% {'(BETTER)' if diff_raw_mse < 0 else '(WORSE)'}")
    report_lines.append(f"{'Baseline-Shifted Pure MSE':<30} | {mean_shf_mse_v3:<20.6f} | {mean_shf_mse_v3_1:<20.6f} | {diff_shf_mse:+.2f}% {'(BETTER)' if diff_shf_mse < 0 else '(WORSE)'}")
    report_lines.append(f"{'Baseline Estimation MSE':<30} | {mean_base_mse_v3:<20.6f} | {mean_base_mse_v3_1:<20.6f} | {diff_base_mse:+.2f}% {'(BETTER)' if diff_base_mse < 0 else '(WORSE)'}")
    report_lines.append(f"{'Spectral Cosine Similarity':<30} | {mean_cos_v3:<20.6f} | {mean_cos_v3_1:<20.6f} | {(mean_cos_v3_1-mean_cos_v3)*100:+.3f}% pts")
    report_lines.append("-" * 100)

    report_lines.append("\n" + "=" * 100)
    report_lines.append(" BREAKDOWN BY SPECTRAL RESOLUTION ")
    report_lines.append("=" * 100)

    # 1. Raw Validation Breakdown
    report_lines.append("\n--- PART A: RAW VALIDATION (PURE MSE) ---")
    report_lines.append(f"{'Resolution':<12} | {'V3.0 Raw MSE':<16} | {'V3.1 Raw MSE':<16} | {'Delta (%)':<15} | {'Winner':<10}")
    report_lines.append("-" * 75)
    for b in BIN_SIZES:
        m_v3 = results[b]["v3"]["raw_mse"]
        m_v3_1 = results[b]["v3_1"]["raw_mse"]
        delta = ((m_v3_1 - m_v3) / m_v3) * 100
        winner = "V3.1" if delta < 0 else "V3.0"
        report_lines.append(f"{b:<12} | {m_v3:<16.6f} | {m_v3_1:<16.6f} | {delta:+14.2f}% | {winner:<10}")

    # 2. Baseline Shifted Breakdown
    report_lines.append("\n--- PART B: BASELINE-SHIFTED VALIDATION (PEAK SHAPE / RECONSTRUCTION MSE) ---")
    report_lines.append(f"{'Resolution':<12} | {'V3.0 Shift MSE':<16} | {'V3.1 Shift MSE':<16} | {'Delta (%)':<15} | {'Winner':<10}")
    report_lines.append("-" * 75)
    for b in BIN_SIZES:
        m_v3 = results[b]["v3"]["shf_mse"]
        m_v3_1 = results[b]["v3_1"]["shf_mse"]
        delta = ((m_v3_1 - m_v3) / m_v3) * 100
        winner = "V3.1" if delta < 0 else "V3.0"
        report_lines.append(f"{b:<12} | {m_v3:<16.6f} | {m_v3_1:<16.6f} | {delta:+14.2f}% | {winner:<10}")

    # 3. Baseline Estimation Accuracy Breakdown
    report_lines.append("\n--- PART C: BASELINE ESTIMATION MSE ---")
    report_lines.append(f"{'Resolution':<12} | {'V3.0 Base MSE':<16} | {'V3.1 Base MSE':<16} | {'Delta (%)':<15} | {'Winner':<10}")
    report_lines.append("-" * 75)
    for b in BIN_SIZES:
        m_v3 = results[b]["v3"]["base_mse"]
        m_v3_1 = results[b]["v3_1"]["base_mse"]
        delta = ((m_v3_1 - m_v3) / m_v3) * 100
        winner = "V3.1" if delta < 0 else "V3.0"
        report_lines.append(f"{b:<12} | {m_v3:<16.6f} | {m_v3_1:<16.6f} | {delta:+14.2f}% | {winner:<10}")

    report_content = "\n".join(report_lines) + "\n"
    print(report_content)

    with open(DEFAULT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nFull Validation Results saved to: {DEFAULT_REPORT_PATH}")

    # Also save the executive summary report
    summary_path = os.path.join(SCRIPT_DIR, "head_to_head_summary.txt")
    summary_lines = [
        "=" * 80,
        " EXECUTIVE SUMMARY: POLYGAUSSNET V3 vs V3.1 HEAD-TO-HEAD (ROUND 7) ",
        "=" * 80,
        f"Total Parameters: V3 = {params_v3:,} | V3.1 = {params_v3_1:,} ({(params_v3_1-params_v3)/params_v3*100:+.1f}%)",
        f"Epochs Trained:   8",
        f"Dataset:          training_round7 (40,960 train / 5,120 test spectra)",
        "-" * 80,
        "OVERALL COMPARISON:",
        f"  * Raw Pure MSE:             V3: {mean_raw_mse_v3:.6f} | V3.1: {mean_raw_mse_v3_1:.6f} ({diff_raw_mse:+.2f}%)",
        f"  * Baseline-Shifted Pure MSE: V3: {mean_shf_mse_v3:.6f} | V3.1: {mean_shf_mse_v3_1:.6f} ({diff_shf_mse:+.2f}%)",
        f"  * Baseline Estimation MSE:   V3: {mean_base_mse_v3:.6f} | V3.1: {mean_base_mse_v3_1:.6f} ({diff_base_mse:+.2f}%)",
        f"  * Spectral Cosine Sim:       V3: {mean_cos_v3:.6f} | V3.1: {mean_cos_v3_1:.6f} ({(mean_cos_v3_1-mean_cos_v3)*100:+.3f}% pts)",
        "-" * 80,
        "RESOLUTION BREAKDOWN (Raw MSE):",
    ]
    for b in BIN_SIZES:
        m_v3 = results[b]["v3"]["raw_mse"]
        m_v3_1 = results[b]["v3_1"]["raw_mse"]
        delta = ((m_v3_1 - m_v3) / m_v3) * 100
        winner = "V3.1" if delta < 0 else "V3.0"
        summary_lines.append(f"  * {b:>4} bins: V3={m_v3:.6f} | V3.1={m_v3_1:.6f} -> {delta:+6.2f}% ({winner})")

    summary_lines.extend([
        "-" * 80,
        "RESOLUTION BREAKDOWN (Shifted MSE):",
    ])
    for b in BIN_SIZES:
        m_v3 = results[b]["v3"]["shf_mse"]
        m_v3_1 = results[b]["v3_1"]["shf_mse"]
        delta = ((m_v3_1 - m_v3) / m_v3) * 100
        winner = "V3.1" if delta < 0 else "V3.0"
        summary_lines.append(f"  * {b:>4} bins: V3={m_v3:.6f} | V3.1={m_v3_1:.6f} -> {delta:+6.2f}% ({winner})")

    summary_lines.append("=" * 80 + "\n")
    summary_content = "\n".join(summary_lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"Executive Summary saved to:       {summary_path}")


if __name__ == "__main__":
    run_benchmark()
