"""
MULTI-SEED BENCHMARK - ROUND 10
-------------------------------
Evaluates every checkpoint found in training_round10/runs/seed_*/ on the fresh held-out
test set (training_round10/testing_data10, 5 x 2,048 spectra).

Models evaluated:
  - v3         : PolyGaussNet V3.0       (Original reference benchmark)
  - v3_1       : PolyGaussNet V3.1       (32-ch trunk, 1-bucket pool, 33->128->64->8 FC)
  - v3_2       : PolyGaussNet V3.2       (32-ch trunk + 1x1 proj to 128, 1-bucket pool, 129->64->8 funnel)
  - v3_3       : PolyGaussNet V3.3       (32-ch trunk, 4-quadrant pool, 129->64->8 funnel, WITH length scalar)
  - v3_3_nolen : PolyGaussNet V3.3-NoLen (Ablation: V3.3 with length scalar REMOVED, 128->64->8 funnel)

Reference model for deltas:
  - Primary benchmark: V3.0 (original baseline)
  - Secondary comparison: V3.1 (direct architectural comparison)
  - Length-scalar ablation: V3.3-NoLen vs V3.3
  - Negative delta (%) is BETTER for MSE metrics.
  - Positive delta (p) is BETTER for Cosine Similarity.

Report contents:
  1. Per-seed table comparing all models (with % change vs V3.0)
  2. Direct head-to-head deltas for V3.2, V3.3 & V3.3-NoLen vs V3.1
  3. Per-resolution performance breakdown across 800 to 1600 bins
  4. Aggregate across seeds: mean +/- std per model, and mean +/- std of deltas vs V3.0
  5. Length-scalar ablation (paired, per seed) with win counts
  6. 16-Epoch vs 8-Epoch progression analysis (if runs_8ep/ snapshots are present)

Usage:
  python testing10.py
"""

import os
import sys
import glob
import shutil

import numpy as np
import torch

# =====================================================================
# RUN SETTINGS (edit these constants)
# =====================================================================
EVALUATE_8EP_SNAPSHOTS = True    # Also evaluate runs_8ep/ snapshots if present

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1
from polygaussnet3_2 import PolyGaussNet as PolyGaussNet3_2
from polygaussnet3_3 import PolyGaussNet as PolyGaussNet3_3
from polygaussnet3_3_nolen import PolyGaussNet as PolyGaussNet3_3_NoLen

MODEL_REGISTRY = {
    "v3": ("V3.0", PolyGaussNet3),
    "v3_1": ("V3.1", PolyGaussNet3_1),
    "v3_2": ("V3.2", PolyGaussNet3_2),
    "v3_3": ("V3.3", PolyGaussNet3_3),
    "v3_3_nolen": ("V3.3-NoLen", PolyGaussNet3_3_NoLen),
}
MODEL_ORDER = ["v3", "v3_1", "v3_2", "v3_3", "v3_3_nolen"]
SECONDARY_VS_V31 = ["v3_2", "v3_3", "v3_3_nolen"]

# Length-scalar ablation pairs: (model, reference, description)
LENGTH_ABLATIONS = [
    ("v3_3_nolen", "v3_3", "REMOVING length scalar from V3.3 (V3.3-NoLen vs V3.3)"),
]

TEST_DIR = os.path.join(SCRIPT_DIR, "testing_data10")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
RUNS_8EP_DIR = os.path.join(SCRIPT_DIR, "runs_8ep")
REPORT_PATH = os.path.join(SCRIPT_DIR, "validation_results_round10.txt")

BIN_SIZES = [800, 1016, 1200, 1400, 1600]
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31

METRICS = [
    ("raw_mse", "Raw Pure MSE", True),
    ("shf_mse", "Baseline-Shifted MSE", True),
    ("base_mse", "Baseline Est. MSE", True),
    ("cos", "Cosine Similarity", False),
]

W = 125  # report width


# =====================================================================
# EVALUATION HELPERS
# =====================================================================
def load_test_chunk(filepath: str):
    with np.load(filepath) as data:
        raw = data["full_matrix"].astype(np.float32)
        pure = data["pure_matrix"].astype(np.float32)
        bc_target = data["pure_noise_cosmic_matrix"].astype(np.float32)
        baseline = data["baseline_matrix"].astype(np.float32) if "baseline_matrix" in data else raw - bc_target
    return raw, pure, baseline


def load_test_bin(files: list[str]):
    """Loads and concatenates all test chunks for one resolution."""
    chunks = [load_test_chunk(f) for f in files]
    return tuple(np.concatenate([c[i] for c in chunks], axis=0) for i in range(3))


def align_shift(pred, target):
    return pred + (np.mean(target, axis=-1, keepdims=True) - np.mean(pred, axis=-1, keepdims=True))


def mse(pred, target):
    return float(np.mean((pred - target) ** 2))


def cosine(pred, target):
    dot = np.sum(pred * target, axis=-1)
    return float(np.mean(dot / ((np.linalg.norm(pred, axis=-1) + 1e-8) * (np.linalg.norm(target, axis=-1) + 1e-8))))


def run_model(model, raw_np, device, batch_size=256):
    model.eval()
    clean_out, base_out = [], []
    with torch.no_grad():
        for i in range(0, raw_np.shape[0], batch_size):
            batch = torch.from_numpy(raw_np[i:i + batch_size]).unsqueeze(1).to(device)
            clean, base, _, _ = model(batch)
            clean_out.append(clean.squeeze(1).cpu().numpy())
            base_out.append(base.squeeze(1).cpu().numpy())
    return np.concatenate(clean_out), np.concatenate(base_out)


def evaluate_checkpoint(key, ckpt_path, test_data, device):
    """Returns ({bins: {metric: value}}, n_params)."""
    _, cls = MODEL_REGISTRY[key]
    model = cls(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    n_params = sum(p.numel() for p in model.parameters())
    out = {}
    for b, (raw, pure, base_true) in test_data.items():
        clean, base = run_model(model, raw, device)
        out[b] = {
            "raw_mse": mse(clean, pure),
            "shf_mse": mse(align_shift(clean, pure), pure),
            "base_mse": mse(base, base_true),
            "cos": cosine(clean, pure),
        }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return out, n_params


def discover_runs(runs_dir: str):
    """Returns {run_name: {model_key: checkpoint_path}}."""
    runs = {}
    for seed_dir in sorted(glob.glob(os.path.join(runs_dir, "seed_*"))):
        ckpts = {k: os.path.join(seed_dir, f"{k}.pth") for k in MODEL_ORDER
                 if os.path.exists(os.path.join(seed_dir, f"{k}.pth"))}
        if ckpts:
            runs[os.path.basename(seed_dir)] = ckpts
    return runs


def pct(new, ref):
    return (new - ref) / ref * 100.0


def delta(m, val, ref):
    return (val - ref) * 100 if m == "cos" else pct(val, ref)


def unit(m):
    return "p" if m == "cos" else "%"


def is_better(m, d):
    return d > 0 if m == "cos" else d < 0


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using Compute Device: {device}")

    runs = discover_runs(RUNS_DIR)
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {RUNS_DIR}. Run train_round10.py first.")
    print(f"Discovered runs: { {r: list(c.keys()) for r, c in runs.items()} }")

    # Load test data once
    test_data = {}
    for b in BIN_SIZES:
        files = sorted(glob.glob(os.path.join(TEST_DIR, str(b), "*.npz")))
        if not files:
            raise FileNotFoundError(f"No test chunk for {b} bins in {TEST_DIR}. Run create_dataset10.py first.")
        test_data[b] = load_test_bin(files)
    n_per_bin = {b: test_data[b][0].shape[0] for b in BIN_SIZES}

    results, per_res, n_params = {}, {}, {}
    for run_name, ckpts in runs.items():
        results[run_name], per_res[run_name] = {}, {}
        for key in MODEL_ORDER:
            if key not in ckpts:
                continue
            print(f"  Evaluating {run_name} / {MODEL_REGISTRY[key][0]} ({ckpts[key]}) ...")
            per_res[run_name][key], n_params[key] = evaluate_checkpoint(key, ckpts[key], test_data, device)
            results[run_name][key] = {
                m: float(np.mean([per_res[run_name][key][b][m] for b in BIN_SIZES])) for m, _, _ in METRICS
            }

    # =================================================================
    # BUILD REPORT
    # =================================================================
    L = []
    L.append("=" * W)
    L.append(" ROUND 10: ARCHITECTURAL BENCHMARK + LENGTH ABLATION (V3.0, V3.1, V3.2, V3.3, V3.3-NoLen at 16 Epochs) ")
    L.append("=" * W)
    n_list = sorted(set(n_per_bin.values()))
    n_str = f"{n_list[0]:,}" if len(n_list) == 1 else str(n_per_bin)
    L.append(f"Test set: {TEST_DIR}  ({len(BIN_SIZES)} resolutions x {n_str} spectra)")
    L.append("Parameters: " + " | ".join(f"{MODEL_REGISTRY[k][0]} = {n_params[k]:,}" for k in MODEL_ORDER if k in n_params))
    L.append(f"Runs: {', '.join(runs.keys())}")
    L.append("Primary Benchmark: V3.0 (original). Cosine deltas are in absolute % pts (p); other deltas are relative % (negative = better).")

    # ---- 1. Per-run tables ----
    for run_name in runs:
        L.append("\n" + "-" * W)
        L.append(f" RUN: {run_name}")
        L.append("-" * W)

        present = [k for k in MODEL_ORDER if k in results[run_name]]
        ref_key = "v3" if "v3" in results[run_name] else ("v3_1" if "v3_1" in results[run_name] else None)
        header = f"{'Metric':<24}" + "".join(f" | {MODEL_REGISTRY[k][0]:>12}" for k in present)
        if ref_key:
            header += "".join(f" | {'Δ ' + MODEL_REGISTRY[k][0] + ' vs ' + MODEL_REGISTRY[ref_key][0]:>22}"
                              for k in present if k != ref_key)
        L.append(header)

        for m, mname, _ in METRICS:
            row = f"{mname:<24}" + "".join(f" | {results[run_name][k][m]:>12.6f}" for k in present)
            if ref_key:
                ref = results[run_name][ref_key][m]
                for k in present:
                    if k == ref_key:
                        continue
                    d = delta(m, results[run_name][k][m], ref)
                    row += f" | {d:>+21.2f}{unit(m)}"
            L.append(row)

        # Secondary comparison vs V3.1
        if "v3_1" in results[run_name] and any(k in results[run_name] for k in SECONDARY_VS_V31):
            L.append(f"\n  Secondary Comparison vs V3.1 ({run_name}):")
            for m, mname, _ in METRICS:
                ref_31 = results[run_name]["v3_1"][m]
                parts = []
                for k in SECONDARY_VS_V31:
                    if k in results[run_name]:
                        d = delta(m, results[run_name][k][m], ref_31)
                        better = " (BETTER)" if is_better(m, d) else " (WORSE)"
                        parts.append(f"{MODEL_REGISTRY[k][0]}: {d:>+7.2f}{unit(m)}{better}")
                L.append(f"    * {mname:<22}: " + " | ".join(parts))

        # Per-resolution raw MSE
        L.append(f"\n  Raw Pure MSE by resolution ({run_name}):")
        L.append("  " + f"{'Bins':<8}" + "".join(f" | {MODEL_REGISTRY[k][0]:>10}" for k in MODEL_ORDER if k in per_res[run_name]))
        for b in BIN_SIZES:
            L.append("  " + f"{b:<8}" + "".join(f" | {per_res[run_name][k][b]['raw_mse']:>10.6f}"
                                                for k in MODEL_ORDER if k in per_res[run_name]))

    # ---- 2. Aggregate across runs ----
    L.append("\n" + "=" * W)
    L.append(" AGGREGATE ACROSS RUNS (mean ± std; deltas computed relative to V3.0 where present)")
    L.append("=" * W)
    for m, mname, _ in METRICS:
        L.append(f"\n{mname}:")
        for k in MODEL_ORDER:
            vals = [results[r][k][m] for r in runs if k in results[r]]
            if not vals:
                continue
            line = f"  {MODEL_REGISTRY[k][0]:<10} n={len(vals)} | {np.mean(vals):.6f} ± {sd(vals):.6f}"
            if k != "v3":
                deltas = [delta(m, results[r][k][m], results[r]["v3"][m]) for r in runs if k in results[r] and "v3" in results[r]]
                if deltas:
                    line += f" | Δ vs V3.0: {np.mean(deltas):+.2f}{unit(m)} ± {sd(deltas):.2f}  (per-run: {', '.join(f'{d:+.1f}' for d in deltas)})"
            if k in SECONDARY_VS_V31:
                deltas_31 = [delta(m, results[r][k][m], results[r]["v3_1"][m]) for r in runs if k in results[r] and "v3_1" in results[r]]
                if deltas_31:
                    line += f" | Δ vs V3.1: {np.mean(deltas_31):+.2f}{unit(m)} ± {sd(deltas_31):.2f}"
            L.append(line)

    # ---- 3. Length-scalar ablation ----
    L.append("\n" + "=" * W)
    L.append(" LENGTH-SCALAR ABLATION (paired per seed: identical seed, data & batch order; only the length input differs)")
    L.append("   Δ = (model - reference). MSE: negative = model better. Cosine: positive = model better.")
    L.append("   'wins' = number of seeds where the model beat its reference.")
    L.append("=" * W)
    for model_k, ref_k, desc in LENGTH_ABLATIONS:
        paired = [r for r in runs if model_k in results[r] and ref_k in results[r]]
        L.append(f"\n{desc}:")
        if not paired:
            L.append("  (no paired runs available)")
            continue
        for m, mname, _ in METRICS:
            ds = [delta(m, results[r][model_k][m], results[r][ref_k][m]) for r in paired]
            wins = sum(is_better(m, d) for d in ds)
            L.append(f"  {mname:<22}: {np.mean(ds):>+7.2f}{unit(m)} ± {sd(ds):>5.2f}  | wins {wins}/{len(ds)}"
                     f"  (per-run: {', '.join(f'{d:+.1f}' for d in ds)})")

    # ---- 4. 8-Epoch Snapshot Comparison (if present) ----
    if EVALUATE_8EP_SNAPSHOTS and os.path.exists(RUNS_8EP_DIR):
        runs_8ep = discover_runs(RUNS_8EP_DIR)
        if runs_8ep:
            L.append("\n" + "=" * W)
            L.append(" 16-EPOCH EXTENDED PROGRESSION (16 Epochs vs 8-Epoch Snapshot) ")
            L.append("=" * W)
            for seed_name, ckpts_8 in runs_8ep.items():
                if seed_name not in results:
                    continue
                keys = [k for k in MODEL_ORDER if k in ckpts_8 and k in results[seed_name]]
                L.append(f"\n--- {seed_name} (16 Epochs vs 8-Epoch Snapshot) ---")
                header = f"{'Metric':<24}"
                for k in keys:
                    lbl = MODEL_REGISTRY[k][0]
                    header += f" | {lbl + ' (8ep)':>16} | {lbl + ' (16ep)':>17} | {'Δ 16v8':>8}"
                L.append(header)

                res_8 = {}
                for k in keys:
                    print(f"  Evaluating 8ep snapshot {seed_name} / {MODEL_REGISTRY[k][0]} ...")
                    res_8[k], _ = evaluate_checkpoint(k, ckpts_8[k], test_data, device)

                mean_8 = {k: {m: float(np.mean([res_8[k][b][m] for b in BIN_SIZES])) for m, _, _ in METRICS} for k in res_8}

                for m, mname, _ in METRICS:
                    row = f"{mname:<24}"
                    for k in keys:
                        val_8 = mean_8[k][m]
                        val_16 = results[seed_name][k][m]
                        d_16v8 = delta(m, val_16, val_8)
                        row += f" | {val_8:>16.6f} | {val_16:>17.6f} | {d_16v8:>+7.2f}{unit(m)}"
                    L.append(row)

    report = "\n".join(L) + "\n"
    print("\n" + report)

    if os.path.exists(REPORT_PATH):
        backup_path = REPORT_PATH.replace(".txt", "_prev.txt")
        shutil.copyfile(REPORT_PATH, backup_path)
        print(f"Archived previous report to: {backup_path}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
