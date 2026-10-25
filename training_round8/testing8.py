"""
MULTI-SEED BENCHMARK - ROUND 8
------------------------------
Evaluates every checkpoint found in training_round8/runs/seed_*/ on the held-out
Round 7 test set (training_round7/testing_data7, 5 x 1,024 spectra), using the exact
same metrics as testing7.py.

Report contents:
  1. Per-seed table for each model (and % change vs V3 of the same seed)
  2. Aggregate across seeds: mean +/- std per model, and mean +/- std of the % change vs V3
  3. Gain attribution: what fraction of the V3 -> V3.1 gain is reproduced by V3-Len
     (length input only). Small fraction => the V3.1 architecture is driving the gain.

INCLUDE_ROUND7 (top of file) adds the Round 7 checkpoints (v3, v3_1) as 'r7_seed_42'.
  Note: Round 7 batch order was not perfectly matched between models (see train_round8.py).

Usage:
  python testing8.py
"""

import os
import sys
import glob
import shutil

import numpy as np
import torch

# =====================================================================
# RUN SETTINGS (edit these)
# =====================================================================
INCLUDE_ROUND7 = True   # Also evaluate the Round 7 (seed 42) V3 / V3.1 checkpoints

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1
from polygaussnet3_len import PolyGaussNet as PolyGaussNet3_Len

MODEL_REGISTRY = {
    "v3": ("V3.0", PolyGaussNet3),
    "v3_len": ("V3-Len", PolyGaussNet3_Len),
    "v3_1": ("V3.1", PolyGaussNet3_1),
}
MODEL_ORDER = ["v3", "v3_len", "v3_1"]

TEST_DIR = os.path.join(PROJECT_ROOT, "training_round7", "testing_data7")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
RUNS_16EP_DIR = os.path.join(SCRIPT_DIR, "runs_16ep")
ROUND7_DIR = os.path.join(PROJECT_ROOT, "training_round7")
REPORT_PATH = os.path.join(SCRIPT_DIR, "validation_results_round8.txt")

BIN_SIZES = [800, 1016, 1200, 1400, 1600]
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31

METRICS = [
    ("raw_mse", "Raw Pure MSE", True),
    ("shf_mse", "Baseline-Shifted MSE", True),
    ("base_mse", "Baseline Est. MSE", True),
    ("cos", "Cosine Similarity", False),
]


# =====================================================================
# EVALUATION HELPERS (identical math to testing7.py)
# =====================================================================
def load_test_chunk(filepath: str):
    with np.load(filepath) as data:
        raw = data["full_matrix"].astype(np.float32)
        pure = data["pure_matrix"].astype(np.float32)
        bc_target = data["pure_noise_cosmic_matrix"].astype(np.float32)
        baseline = data["baseline_matrix"].astype(np.float32) if "baseline_matrix" in data else raw - bc_target
    return raw, pure, baseline


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


def discover_runs(include_round7: bool):
    """Returns {run_name: {model_key: checkpoint_path}}."""
    runs = {}
    for seed_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "seed_*"))):
        ckpts = {k: os.path.join(seed_dir, f"{k}.pth") for k in MODEL_ORDER
                 if os.path.exists(os.path.join(seed_dir, f"{k}.pth"))}
        if ckpts:
            runs[os.path.basename(seed_dir)] = ckpts
    if include_round7:
        r7 = {}
        for key, fname in [("v3", "polygaussnet3.pth"), ("v3_1", "polygaussnet3_1.pth")]:
            p = os.path.join(ROUND7_DIR, fname)
            if os.path.exists(p):
                r7[key] = p
        if r7:
            runs["r7_seed_42"] = r7
    return runs


def pct(new, ref):
    return (new - ref) / ref * 100.0


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using Compute Device: {device}")

    runs = discover_runs(INCLUDE_ROUND7)
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {RUNS_DIR}. Run train_round8.py first.")
    print(f"Discovered runs: { {r: list(c.keys()) for r, c in runs.items()} }")

    # Load test data once
    test_data = {}
    for b in BIN_SIZES:
        files = sorted(glob.glob(os.path.join(TEST_DIR, str(b), "*.npz")))
        if not files:
            raise FileNotFoundError(f"No test chunk for {b} bins in {TEST_DIR}")
        test_data[b] = load_test_chunk(files[0])

    # results[run][model][metric] = mean over resolutions; per_res[run][model][bin][metric]
    results, per_res, n_params = {}, {}, {}
    for run_name, ckpts in runs.items():
        results[run_name], per_res[run_name] = {}, {}
        for key in MODEL_ORDER:
            if key not in ckpts:
                continue
            label, cls = MODEL_REGISTRY[key]
            model = cls(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
            model.load_state_dict(torch.load(ckpts[key], map_location=device))
            n_params[key] = sum(p.numel() for p in model.parameters())
            print(f"  Evaluating {run_name} / {label} ...")

            per_res[run_name][key] = {}
            for b, (raw, pure, base_true) in test_data.items():
                clean, base = run_model(model, raw, device)
                per_res[run_name][key][b] = {
                    "raw_mse": mse(clean, pure),
                    "shf_mse": mse(align_shift(clean, pure), pure),
                    "base_mse": mse(base, base_true),
                    "cos": cosine(clean, pure),
                }
            results[run_name][key] = {
                m: float(np.mean([per_res[run_name][key][b][m] for b in BIN_SIZES])) for m, _, _ in METRICS
            }
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    # =================================================================
    # BUILD REPORT
    # =================================================================
    L = []
    L.append("=" * 100)
    L.append(" ROUND 8: MULTI-SEED VALIDATION (V3.0 vs V3-Len vs V3.1) ")
    L.append("=" * 100)
    L.append(f"Test set: {TEST_DIR}  ({len(BIN_SIZES)} resolutions x 1,024 spectra)")
    L.append("Parameters: " + " | ".join(f"{MODEL_REGISTRY[k][0]} = {n_params[k]:,}" for k in MODEL_ORDER if k in n_params))
    L.append(f"Runs: {', '.join(runs.keys())}")
    L.append("Cosine deltas are reported in absolute % points; all other deltas are relative % vs V3.0 (negative = better).")

    # ---- 1. Per-run tables ----
    for run_name in runs:
        L.append("\n" + "-" * 100)
        L.append(f" RUN: {run_name}")
        L.append("-" * 100)
        header = f"{'Metric':<24}" + "".join(f" | {MODEL_REGISTRY[k][0]:>12}" for k in MODEL_ORDER if k in results[run_name])
        header += "".join(f" | {'Δ ' + MODEL_REGISTRY[k][0]:>12}" for k in MODEL_ORDER if k != "v3" and k in results[run_name])
        L.append(header)
        for m, mname, _ in METRICS:
            row = f"{mname:<24}" + "".join(f" | {results[run_name][k][m]:>12.6f}" for k in MODEL_ORDER if k in results[run_name])
            if "v3" in results[run_name]:
                ref = results[run_name]["v3"][m]
                for k in MODEL_ORDER:
                    if k == "v3" or k not in results[run_name]:
                        continue
                    val = results[run_name][k][m]
                    d = (val - ref) * 100 if m == "cos" else pct(val, ref)
                    row += f" | {d:>+11.2f}{'p' if m == 'cos' else '%'}"
            L.append(row)

        # Per-resolution raw MSE
        L.append(f"\n  Raw Pure MSE by resolution ({run_name}):")
        L.append("  " + f"{'Bins':<8}" + "".join(f" | {MODEL_REGISTRY[k][0]:>10}" for k in MODEL_ORDER if k in per_res[run_name]))
        for b in BIN_SIZES:
            L.append("  " + f"{b:<8}" + "".join(f" | {per_res[run_name][k][b]['raw_mse']:>10.6f}"
                                                for k in MODEL_ORDER if k in per_res[run_name]))

    # ---- 2. Aggregate across runs ----
    L.append("\n" + "=" * 100)
    L.append(" AGGREGATE ACROSS RUNS (mean ± std; only runs containing V3.0 are used for deltas)")
    L.append("=" * 100)
    for m, mname, _ in METRICS:
        L.append(f"\n{mname}:")
        for k in MODEL_ORDER:
            vals = [results[r][k][m] for r in runs if k in results[r]]
            if not vals:
                continue
            line = f"  {MODEL_REGISTRY[k][0]:<8} n={len(vals)} | {np.mean(vals):.6f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.6f}"
            if k != "v3":
                deltas = [((results[r][k][m] - results[r]["v3"][m]) * 100 if m == "cos" else pct(results[r][k][m], results[r]["v3"][m]))
                          for r in runs if k in results[r] and "v3" in results[r]]
                if deltas:
                    unit = "p" if m == "cos" else "%"
                    sd = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
                    line += f" | Δ vs V3.0: {np.mean(deltas):+.2f}{unit} ± {sd:.2f}  (per-run: {', '.join(f'{d:+.1f}' for d in deltas)})"
            L.append(line)

    # ---- 3. Gain attribution ----
    L.append("\n" + "=" * 100)
    L.append(" GAIN ATTRIBUTION: share of the V3.0 -> V3.1 gain reproduced by length input alone (V3-Len)")
    L.append("   share = (V3.0 - V3-Len) / (V3.0 - V3.1)   [error metrics; for cosine the sign is flipped]")
    L.append("   ~0%  => length input contributes nothing; ~100% => length input explains the whole gain")
    L.append("=" * 100)
    attrib_runs = [r for r in runs if all(k in results[r] for k in MODEL_ORDER)]
    if not attrib_runs:
        L.append("  (No run contains all three models yet.)")
    else:
        for m, mname, lower_better in METRICS:
            shares = []
            for r in attrib_runs:
                v3, vl, v31 = (results[r][k][m] for k in MODEL_ORDER)
                total_gain = (v3 - v31) if lower_better else (v31 - v3)
                len_gain = (v3 - vl) if lower_better else (vl - v3)
                shares.append(len_gain / total_gain * 100 if total_gain != 0 else float("nan"))
            sd = np.std(shares, ddof=1) if len(shares) > 1 else 0.0
            L.append(f"  {mname:<24}: {np.nanmean(shares):6.1f}% ± {sd:5.1f}  (per-run: {', '.join(f'{s:.1f}%' for s in shares)})")

    # ---- 4. 16-Epoch Extended Runs (if present) ----
    runs_16ep = {}
    if os.path.exists(RUNS_16EP_DIR):
        for seed_dir in sorted(glob.glob(os.path.join(RUNS_16EP_DIR, "seed_*"))):
            ckpts = {k: os.path.join(seed_dir, f"{k}.pth") for k in MODEL_ORDER
                     if os.path.exists(os.path.join(seed_dir, f"{k}.pth"))}
            if ckpts:
                runs_16ep[os.path.basename(seed_dir)] = ckpts

    if runs_16ep:
        L.append("\n" + "=" * 100)
        L.append(" 16-EPOCH EXTENDED BENCHMARK (Comparing 16 Epochs vs 8-Epoch Snapshot) ")
        L.append("=" * 100)
        for seed_name, ckpts_16 in runs_16ep.items():
            L.append(f"\n--- {seed_name} (16 Epochs vs 8-Epoch Snapshot) ---")
            header = f"{'Metric':<24}"
            for k in MODEL_ORDER:
                if k in ckpts_16:
                    header += f" | {MODEL_REGISTRY[k][0]+' (8ep)':>12} | {MODEL_REGISTRY[k][0]+' (16ep)':>13} | {'Δ 16v8':>8}"
            L.append(header)

            res_16 = {}
            for k in MODEL_ORDER:
                if k not in ckpts_16:
                    continue
                label, cls = MODEL_REGISTRY[k]
                m_obj = cls(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
                m_obj.load_state_dict(torch.load(ckpts_16[k], map_location=device))
                res_16[k] = {}
                for b, (raw, pure, base_true) in test_data.items():
                    clean, base = run_model(m_obj, raw, device)
                    res_16[k][b] = {
                        "raw_mse": mse(clean, pure),
                        "shf_mse": mse(align_shift(clean, pure), pure),
                        "base_mse": mse(base, base_true),
                        "cos": cosine(clean, pure),
                    }
                del m_obj
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            mean_16 = {k: {m: float(np.mean([res_16[k][b][m] for b in BIN_SIZES])) for m, _, _ in METRICS} for k in res_16}

            for m, mname, _ in METRICS:
                row = f"{mname:<24}"
                for k in MODEL_ORDER:
                    if k not in ckpts_16:
                        continue
                    val_16 = mean_16[k][m]
                    val_8 = results.get(seed_name, {}).get(k, {}).get(m, None)
                    if val_8 is not None:
                        d_16v8 = (val_16 - val_8) * 100 if m == "cos" else pct(val_16, val_8)
                        row += f" | {val_8:>12.6f} | {val_16:>13.6f} | {d_16v8:>+7.2f}{'p' if m == 'cos' else '%'}"
                    else:
                        row += f" | {'N/A':>12} | {val_16:>13.6f} | {'N/A':>8}"
                L.append(row)

            if "v3" in mean_16 and "v3_1" in mean_16:
                L.append(f"\n  Head-to-head at 16 epochs ({seed_name}):")
                for m, mname, _ in METRICS:
                    v3_val = mean_16["v3"][m]
                    v31_val = mean_16["v3_1"][m]
                    d = (v31_val - v3_val) * 100 if m == "cos" else pct(v31_val, v3_val)
                    better = " (BETTER)" if (d < 0 if m != "cos" else d > 0) else " (WORSE)"
                    L.append(f"    * {mname:<22}: V3={v3_val:.6f} | V3.1={v31_val:.6f} -> Δ V3.1: {d:>+7.2f}{'p' if m == 'cos' else '%'}{better}")

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
