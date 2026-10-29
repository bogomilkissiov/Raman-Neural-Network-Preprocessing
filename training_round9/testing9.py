"""
MULTI-SEED BENCHMARK - ROUND 9
------------------------------
Evaluates every checkpoint found in training_round9/runs/seed_*/ on the held-out
test set (training_round7/testing_data7, 5 x 1,024 spectra).

Models evaluated:
  - v3   : PolyGaussNet V3.0 (Original reference benchmark)
  - v3_1 : PolyGaussNet V3.1 (Intermediate model: 32-ch trunk, 1-bucket pool, 33->128->64->8 FC)
  - v3_2 : PolyGaussNet V3.2 (Widened 128-ch trunk, 1-bucket pool, 129->64->8 contracting funnel)
  - v3_3 : PolyGaussNet V3.3 (Lightweight 32-ch trunk, 4-quadrant pool, 129->64->8 contracting funnel)

Reference model for deltas:
  - Primary benchmark: V3.0 (original baseline)
  - Secondary comparison: V3.1 (direct architectural comparison)
  - Negative delta (%) is BETTER for MSE metrics.
  - Positive delta (p) is BETTER for Cosine Similarity.

Report contents:
  1. Per-seed table comparing V3.0, V3.1, V3.2, V3.3 (with % change vs V3.0)
  2. Direct head-to-head deltas for V3.2 & V3.3 vs V3.1
  3. Per-resolution performance breakdown across 800 to 1600 bins
  4. Aggregate across seeds: mean +/- std per model, and mean +/- std of deltas vs V3.0
  5. 16-Epoch vs 8-Epoch progression analysis (if runs_8ep/ snapshots are present)

Usage:
  python testing9.py
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
REUSE_R8_16EP = True             # Allow using Round 8's 16-ep seed_202 checkpoints (v3, v3_1) if not in Round 9 yet
INCLUDE_R8_16EP = True           # Include Round 8's 16-ep seed_202 as a benchmark run if Round 9 seed_202 is missing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1
from polygaussnet3_2 import PolyGaussNet as PolyGaussNet3_2
from polygaussnet3_3 import PolyGaussNet as PolyGaussNet3_3

MODEL_REGISTRY = {
    "v3": ("V3.0", PolyGaussNet3),
    "v3_1": ("V3.1", PolyGaussNet3_1),
    "v3_2": ("V3.2", PolyGaussNet3_2),
    "v3_3": ("V3.3", PolyGaussNet3_3),
}
MODEL_ORDER = ["v3", "v3_1", "v3_2", "v3_3"]

TEST_DIR = os.path.join(PROJECT_ROOT, "training_round7", "testing_data7")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
RUNS_8EP_DIR = os.path.join(SCRIPT_DIR, "runs_8ep")
R8_16EP_DIR = os.path.join(PROJECT_ROOT, "training_round8", "runs_16ep")
REPORT_PATH = os.path.join(SCRIPT_DIR, "validation_results_round9.txt")

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
# EVALUATION HELPERS
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


def discover_runs():
    """Returns {run_name: {model_key: checkpoint_path}}."""
    runs = {}
    for seed_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "seed_*"))):
        seed_name = os.path.basename(seed_dir)
        ckpts = {k: os.path.join(seed_dir, f"{k}.pth") for k in MODEL_ORDER
                 if os.path.exists(os.path.join(seed_dir, f"{k}.pth"))}

        # Fallback for seed_202 v3 and v3_1 from round 8 if allowed and missing in round 9
        if REUSE_R8_16EP and seed_name == "seed_202":
            for k in ["v3", "v3_1"]:
                if k not in ckpts:
                    r8_p = os.path.join(R8_16EP_DIR, "seed_202", f"{k}.pth")
                    if os.path.exists(r8_p):
                        ckpts[k] = r8_p

        if ckpts:
            runs[seed_name] = ckpts

    # If no runs found in runs/ yet, optionally allow evaluating the Round 8 16-ep checkpoints as a fallback dry run
    if INCLUDE_R8_16EP and not runs:
        r8_ckpts = {k: os.path.join(R8_16EP_DIR, "seed_202", f"{k}.pth") for k in ["v3", "v3_1"]
                    if os.path.exists(os.path.join(R8_16EP_DIR, "seed_202", f"{k}.pth"))}
        if r8_ckpts:
            runs["r8_16ep_seed_202"] = r8_ckpts

    return runs


def pct(new, ref):
    return (new - ref) / ref * 100.0


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using Compute Device: {device}")

    runs = discover_runs()
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {RUNS_DIR}. Run train_round9.py first.")
    print(f"Discovered runs: { {r: list(c.keys()) for r, c in runs.items()} }")

    # Load test data once
    test_data = {}
    for b in BIN_SIZES:
        files = sorted(glob.glob(os.path.join(TEST_DIR, str(b), "*.npz")))
        if not files:
            raise FileNotFoundError(f"No test chunk for {b} bins in {TEST_DIR}")
        test_data[b] = load_test_chunk(files[0])

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
            print(f"  Evaluating {run_name} / {label} ({ckpts[key]}) ...")

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
    L.append("=" * 115)
    L.append(" ROUND 9: ARCHITECTURAL BENCHMARK (V3.0 vs V3.1 vs V3.2 vs V3.3 at 16 Epochs) ")
    L.append("=" * 115)
    L.append(f"Test set: {TEST_DIR}  ({len(BIN_SIZES)} resolutions x 1,024 spectra)")
    L.append("Parameters: " + " | ".join(f"{MODEL_REGISTRY[k][0]} = {n_params[k]:,}" for k in MODEL_ORDER if k in n_params))
    L.append(f"Runs: {', '.join(runs.keys())}")
    L.append("Primary Benchmark: V3.0 (original). Cosine deltas are in absolute % pts (p); other deltas are relative % (negative = better).")

    # ---- 1. Per-run tables ----
    for run_name in runs:
        L.append("\n" + "-" * 115)
        L.append(f" RUN: {run_name}")
        L.append("-" * 115)

        # Primary reference: prefer 'v3', fallback to 'v3_1'
        ref_key = "v3" if "v3" in results[run_name] else ("v3_1" if "v3_1" in results[run_name] else None)
        header = f"{'Metric':<24}" + "".join(f" | {MODEL_REGISTRY[k][0]:>12}" for k in MODEL_ORDER if k in results[run_name])
        if ref_key:
            header += "".join(f" | {'Δ ' + MODEL_REGISTRY[k][0] + ' vs ' + MODEL_REGISTRY[ref_key][0]:>18}"
                              for k in MODEL_ORDER if k != ref_key and k in results[run_name])
        L.append(header)

        for m, mname, _ in METRICS:
            row = f"{mname:<24}" + "".join(f" | {results[run_name][k][m]:>12.6f}" for k in MODEL_ORDER if k in results[run_name])
            if ref_key:
                ref = results[run_name][ref_key][m]
                for k in MODEL_ORDER:
                    if k == ref_key or k not in results[run_name]:
                        continue
                    val = results[run_name][k][m]
                    d = (val - ref) * 100 if m == "cos" else pct(val, ref)
                    row += f" | {d:>+17.2f}{'p' if m == 'cos' else '%'}"
            L.append(row)

        # Secondary comparison vs V3.1 (if both v3_1 and newer versions are present)
        if "v3_1" in results[run_name] and any(k in results[run_name] for k in ["v3_2", "v3_3"]):
            L.append(f"\n  Secondary Comparison vs V3.1 ({run_name}):")
            for m, mname, _ in METRICS:
                ref_31 = results[run_name]["v3_1"][m]
                parts = []
                for k in ["v3_2", "v3_3"]:
                    if k in results[run_name]:
                        val = results[run_name][k][m]
                        d = (val - ref_31) * 100 if m == "cos" else pct(val, ref_31)
                        better = " (BETTER)" if (d < 0 if m != "cos" else d > 0) else " (WORSE)"
                        parts.append(f"{MODEL_REGISTRY[k][0]}: {d:>+7.2f}{'p' if m == 'cos' else '%'}{better}")
                L.append(f"    * {mname:<22}: " + " | ".join(parts))

        # Per-resolution raw MSE
        L.append(f"\n  Raw Pure MSE by resolution ({run_name}):")
        L.append("  " + f"{'Bins':<8}" + "".join(f" | {MODEL_REGISTRY[k][0]:>10}" for k in MODEL_ORDER if k in per_res[run_name]))
        for b in BIN_SIZES:
            L.append("  " + f"{b:<8}" + "".join(f" | {per_res[run_name][k][b]['raw_mse']:>10.6f}"
                                                for k in MODEL_ORDER if k in per_res[run_name]))

    # ---- 2. Aggregate across runs ----
    L.append("\n" + "=" * 115)
    L.append(" AGGREGATE ACROSS RUNS (mean ± std; deltas computed relative to V3.0 where present)")
    L.append("=" * 115)
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
            if k in ["v3_2", "v3_3"]:
                deltas_31 = [((results[r][k][m] - results[r]["v3_1"][m]) * 100 if m == "cos" else pct(results[r][k][m], results[r]["v3_1"][m]))
                             for r in runs if k in results[r] and "v3_1" in results[r]]
                if deltas_31:
                    unit = "p" if m == "cos" else "%"
                    sd_31 = np.std(deltas_31, ddof=1) if len(deltas_31) > 1 else 0.0
                    line += f" | Δ vs V3.1: {np.mean(deltas_31):+.2f}{unit} ± {sd_31:.2f}"
            L.append(line)

    # ---- 3. 8-Epoch Snapshot Comparison (if present) ----
    if EVALUATE_8EP_SNAPSHOTS and os.path.exists(RUNS_8EP_DIR):
        runs_8ep = {}
        for seed_dir in sorted(glob.glob(os.path.join(RUNS_8EP_DIR, "seed_*"))):
            ckpts = {k: os.path.join(seed_dir, f"{k}.pth") for k in MODEL_ORDER
                     if os.path.exists(os.path.join(seed_dir, f"{k}.pth"))}
            if ckpts:
                runs_8ep[os.path.basename(seed_dir)] = ckpts

        if runs_8ep:
            L.append("\n" + "=" * 115)
            L.append(" 16-EPOCH EXTENDED PROGRESSION (16 Epochs vs 8-Epoch Snapshot) ")
            L.append("=" * 115)
            for seed_name, ckpts_8 in runs_8ep.items():
                if seed_name not in results:
                    continue
                L.append(f"\n--- {seed_name} (16 Epochs vs 8-Epoch Snapshot) ---")
                header = f"{'Metric':<24}"
                for k in MODEL_ORDER:
                    if k in ckpts_8 and k in results[seed_name]:
                        header += f" | {MODEL_REGISTRY[k][0]+' (8ep)':>12} | {MODEL_REGISTRY[k][0]+' (16ep)':>13} | {'Δ 16v8':>8}"
                L.append(header)

                res_8 = {}
                for k in MODEL_ORDER:
                    if k not in ckpts_8:
                        continue
                    label, cls = MODEL_REGISTRY[k]
                    m_obj = cls(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
                    m_obj.load_state_dict(torch.load(ckpts_8[k], map_location=device))
                    res_8[k] = {}
                    for b, (raw, pure, base_true) in test_data.items():
                        clean, base = run_model(m_obj, raw, device)
                        res_8[k][b] = {
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

                mean_8 = {k: {m: float(np.mean([res_8[k][b][m] for b in BIN_SIZES])) for m, _, _ in METRICS} for k in res_8}

                for m, mname, _ in METRICS:
                    row = f"{mname:<24}"
                    for k in MODEL_ORDER:
                        if k not in ckpts_8 or k not in results[seed_name]:
                            continue
                        val_8 = mean_8[k][m]
                        val_16 = results[seed_name][k][m]
                        d_16v8 = (val_16 - val_8) * 100 if m == "cos" else pct(val_16, val_8)
                        row += f" | {val_8:>12.6f} | {val_16:>13.6f} | {d_16v8:>+7.2f}{'p' if m == 'cos' else '%'}"
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
