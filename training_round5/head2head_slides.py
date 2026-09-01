"""
INTERACTIVE HEAD-TO-HEAD SLIDESHOW - ROUND 5
--------------------------------------------
Interactive Matplotlib viewer for inspecting 64 test spectra side-by-side:
  - Column 1: Raw Input vs. Inferred Baselines (P2 yellow dashed, P3 aqua dotted)
  - Column 2: Reconstructed Clean Spectra vs. Ground Truth Pure (P2 orange, P3 green, GT beige)
  - Column 3: Error Residuals & MAE comparison (P3 vs. P2)

Controls:
  - [▶ / Right / d / Space] : Next Page / Spectrum
  - [◀ / Left  / a        ] : Previous Page / Spectrum
  - [Tab / m              ] : Toggle between 4-Spectrum Grid View and Single Spectrum View
  - [r                    ] : Jump to Random Page / Spectrum
  - [Esc / q              ] : Exit Slideshow
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# =====================================================================
# PATH RESOLUTION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from polygaussnet2 import PolyGaussNet as PolyGaussNet2
from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from training_round5.head2head_testing import load_test_dataset, run_model_inference, GRUVBOX

# Default Configurations
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "training_round4", "test_data4")
DEFAULT_MODEL_P2 = os.path.join(SCRIPT_DIR, "head2head_polygaussnet2.pth")
DEFAULT_MODEL_P3 = os.path.join(SCRIPT_DIR, "head2head_polygaussnet3.pth")
NUM_SAMPLES = 64
PER_PAGE = 4


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Head-to-Head Slideshow: PolyGaussNet2 vs. PolyGaussNet3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to test dataset directory containing .npz files")
    parser.add_argument("--model-p2", type=str, default=DEFAULT_MODEL_P2,
                        help="Path to PolyGaussNet2 weights")
    parser.add_argument("--model-p3", type=str, default=DEFAULT_MODEL_P3,
                        help="Path to PolyGaussNet3 weights")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Total number of spectra to load in slideshow")
    parser.add_argument("--per-page", type=int, default=PER_PAGE,
                        help="Number of spectra displayed per page in grid view")
    parser.add_argument("--poly-order", type=int, default=7,
                        help="Polynomial degree for baseline head")
    parser.add_argument("--filter-kernel-size", type=int, default=31,
                        help="Filter kernel size")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for selecting diverse test spectra")
    return parser.parse_args()


# =====================================================================
# INTERACTIVE SLIDESHOW CLASS
# =====================================================================
class Head2HeadSlideshow:
    def __init__(self, raw_mat, pure_mat, res_p2, res_p3, sample_indices, per_page=4):
        self.raw = raw_mat
        self.pure = pure_mat
        self.res_p2 = res_p2
        self.res_p3 = res_p3
        self.sample_indices = sample_indices
        self.total_samples = len(raw_mat)
        self.per_page = per_page
        self.total_pages = int(np.ceil(self.total_samples / self.per_page))
        self.current_page = 0
        self.single_mode = False
        self.current_single_idx = 0
        self.wavenumbers = np.arange(raw_mat.shape[1])

        # Configure Gruvbox Dark aesthetic
        plt.rcParams.update({
            "figure.facecolor": GRUVBOX["bg0"],
            "axes.facecolor": GRUVBOX["bg0_hard"],
            "axes.edgecolor": GRUVBOX["bg3"],
            "axes.linewidth": 1.0,
            "axes.labelcolor": GRUVBOX["fg"],
            "axes.titlecolor": GRUVBOX["fg0"],
            "xtick.color": GRUVBOX["fg4"],
            "ytick.color": GRUVBOX["fg4"],
            "grid.color": GRUVBOX["bg2"],
            "grid.alpha": 0.5,
            "grid.linestyle": ":",
            "text.color": GRUVBOX["fg"],
            "legend.facecolor": GRUVBOX["bg0_soft"],
            "legend.edgecolor": GRUVBOX["bg3"],
            "legend.labelcolor": GRUVBOX["fg1"],
            "legend.framealpha": 0.92,
        })

        self.fig = plt.figure(figsize=(18, 11), dpi=110)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def show(self):
        self.update_plot()
        plt.show()

    def update_plot(self):
        self.fig.clf()

        if not self.single_mode:
            # -------------------------------------------------------------
            # GRID VIEW (4 Spectra per page, 3 columns each)
            # -------------------------------------------------------------
            start_i = self.current_page * self.per_page
            end_i = min(start_i + self.per_page, self.total_samples)
            num_rows = end_i - start_i

            axes = self.fig.subplots(num_rows, 3, sharex=True)
            if num_rows == 1:
                axes = np.expand_dims(axes, 0)

            for r_idx, s_idx in enumerate(range(start_i, end_i)):
                raw = self.raw[s_idx]
                pure = self.pure[s_idx]
                clean_2 = self.res_p2["clean"][s_idx]
                base_2 = self.res_p2["baseline"][s_idx]
                clean_3 = self.res_p3["clean"][s_idx]
                base_3 = self.res_p3["baseline"][s_idx]
                global_idx = self.sample_indices[s_idx]

                # Col 1: Raw & Baselines
                ax0 = axes[r_idx, 0]
                ax0.plot(self.wavenumbers, raw, color=GRUVBOX["fg3"], alpha=0.5, label="Raw Input", lw=1.0)
                ax0.plot(self.wavenumbers, base_2, color=GRUVBOX["yellow"], linestyle="--", label="P2 Baseline (Degree 7)", lw=1.3)
                ax0.plot(self.wavenumbers, base_3, color=GRUVBOX["aqua"], linestyle=":", label="P3 Baseline (Degree 7)", lw=1.3)
                ax0.set_title(f"Sample #{global_idx} (Item {s_idx + 1}/{self.total_samples}): Raw & Baselines", fontsize=10, fontweight="bold")
                ax0.legend(loc="upper right", fontsize=7.5)
                ax0.grid(True)

                # Col 2: Preprocessed vs Ground Truth
                ax1 = axes[r_idx, 1]
                ax1.plot(self.wavenumbers, pure, color=GRUVBOX["fg0"], label="Ground Truth Pure", lw=1.5)
                ax1.plot(self.wavenumbers, clean_2, color=GRUVBOX["orange"], alpha=0.85, label="PolyGaussNet2 (No Amp)", lw=1.2)
                ax1.plot(self.wavenumbers, clean_3, color=GRUVBOX["green"], alpha=0.85, label="PolyGaussNet3 (Bounded Amp)", lw=1.2)
                ax1.set_title(f"Sample #{global_idx}: Preprocessed Comparison", fontsize=10, fontweight="bold")
                ax1.legend(loc="upper right", fontsize=7.5)
                ax1.grid(True)

                # Col 3: Residuals
                ax2 = axes[r_idx, 2]
                res_2 = clean_2 - pure
                res_3 = clean_3 - pure
                mae_2 = np.mean(np.abs(res_2))
                mae_3 = np.mean(np.abs(res_3))
                ax2.axhline(0, color=GRUVBOX["gray"], linestyle="--", lw=0.8)
                ax2.plot(self.wavenumbers, res_2, color=GRUVBOX["orange"], alpha=0.75, label=f"P2 Residual (MAE: {mae_2:.4f})", lw=1.0)
                ax2.plot(self.wavenumbers, res_3, color=GRUVBOX["green"], alpha=0.85, label=f"P3 Residual (MAE: {mae_3:.4f})", lw=1.0)
                ax2.set_title(f"Sample #{global_idx}: Residuals", fontsize=10, fontweight="bold")
                ax2.legend(loc="upper right", fontsize=7.5)
                ax2.grid(True)

            axes[-1, 0].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=9)
            axes[-1, 1].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=9)
            axes[-1, 2].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=9)

            self.fig.suptitle(
                f"Head-to-Head Slideshow: Page {self.current_page + 1}/{self.total_pages} (Spectra {start_i + 1}–{end_i} of {self.total_samples}) | "
                f"[◀ / ▶] Navigate Pages | [Tab] Single View | [r] Random | [q] Exit",
                fontsize=11, fontweight="bold", color=GRUVBOX["fg0"], y=0.99
            )

        else:
            # -------------------------------------------------------------
            # SINGLE DETAILED SPECTRUM VIEW (1 Spectrum enlarged)
            # -------------------------------------------------------------
            s_idx = self.current_single_idx
            global_idx = self.sample_indices[s_idx]
            raw = self.raw[s_idx]
            pure = self.pure[s_idx]
            clean_2 = self.res_p2["clean"][s_idx]
            base_2 = self.res_p2["baseline"][s_idx]
            clean_3 = self.res_p3["clean"][s_idx]
            base_3 = self.res_p3["baseline"][s_idx]

            axes = self.fig.subplots(3, 1, sharex=True)

            # Panel 1: Raw & Baselines
            ax0 = axes[0]
            ax0.plot(self.wavenumbers, raw, color=GRUVBOX["fg3"], alpha=0.55, label="Raw Input", lw=1.2)
            ax0.plot(self.wavenumbers, base_2, color=GRUVBOX["yellow"], linestyle="--", label="P2 Baseline (Degree 7)", lw=1.8)
            ax0.plot(self.wavenumbers, base_3, color=GRUVBOX["aqua"], linestyle=":", label="P3 Baseline (Degree 7)", lw=1.8)
            ax0.set_title(f"Detailed Spectrum #{global_idx} (Item {s_idx + 1}/{self.total_samples}): Raw & Estimated Baselines", fontsize=12, fontweight="bold")
            ax0.legend(loc="upper right", fontsize=9)
            ax0.grid(True)

            # Panel 2: Clean vs Ground Truth
            ax1 = axes[1]
            ax1.plot(self.wavenumbers, pure, color=GRUVBOX["fg0"], label="Ground Truth Pure", lw=2.0)
            ax1.plot(self.wavenumbers, clean_2, color=GRUVBOX["orange"], alpha=0.85, label="PolyGaussNet2 (No Amp)", lw=1.5)
            ax1.plot(self.wavenumbers, clean_3, color=GRUVBOX["green"], alpha=0.85, label="PolyGaussNet3 (Bounded Amp)", lw=1.5)
            ax1.set_title(f"Detailed Spectrum #{global_idx}: Clean Reconstructed Spectra", fontsize=12, fontweight="bold")
            ax1.legend(loc="upper right", fontsize=9)
            ax1.grid(True)

            # Panel 3: Residuals
            ax2 = axes[2]
            res_2 = clean_2 - pure
            res_3 = clean_3 - pure
            mae_2 = np.mean(np.abs(res_2))
            mae_3 = np.mean(np.abs(res_3))
            ax2.axhline(0, color=GRUVBOX["gray"], linestyle="--", lw=1.0)
            ax2.plot(self.wavenumbers, res_2, color=GRUVBOX["orange"], alpha=0.8, label=f"P2 Residual (MAE: {mae_2:.5f})", lw=1.2)
            ax2.plot(self.wavenumbers, res_3, color=GRUVBOX["green"], alpha=0.85, label=f"P3 Residual (MAE: {mae_3:.5f})", lw=1.2)
            ax2.set_title(f"Detailed Spectrum #{global_idx}: Error Residuals", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=11)
            ax2.legend(loc="upper right", fontsize=9)
            ax2.grid(True)

            self.fig.suptitle(
                f"Detailed Spectrum {s_idx + 1}/{self.total_samples} (Test #{global_idx}) | "
                f"[◀ / ▶] Navigate | [Tab] 4-Grid View | [r] Random | [q] Exit",
                fontsize=12, fontweight="bold", color=GRUVBOX["fg0"], y=0.99
            )

        self.fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ["right", "d", " ", "down"]:
            if self.single_mode:
                self.current_single_idx = (self.current_single_idx + 1) % self.total_samples
            else:
                self.current_page = (self.current_page + 1) % self.total_pages
            self.update_plot()

        elif event.key in ["left", "a", "up"]:
            if self.single_mode:
                self.current_single_idx = (self.current_single_idx - 1) % self.total_samples
            else:
                self.current_page = (self.current_page - 1) % self.total_pages
            self.update_plot()

        elif event.key in ["tab", "m"]:
            self.single_mode = not self.single_mode
            if self.single_mode:
                self.current_single_idx = self.current_page * self.per_page
            else:
                self.current_page = self.current_single_idx // self.per_page
            self.update_plot()

        elif event.key in ["r"]:
            if self.single_mode:
                self.current_single_idx = np.random.randint(0, self.total_samples)
            else:
                self.current_page = np.random.randint(0, self.total_pages)
            self.update_plot()

        elif event.key in ["escape", "q"]:
            plt.close(self.fig)


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print(" LAUNCHING HEAD-TO-HEAD INTERACTIVE SLIDESHOW (64 SPECTRA) ")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. Load test dataset
    raw_all, pure_all, bc_true_all, _ = load_test_dataset(args.data_dir)
    total_available = len(raw_all)

    # 2. Select 64 diverse sample indices across dataset chunks
    rng = np.random.default_rng(seed=args.random_seed)
    num_to_select = min(args.num_samples, total_available)
    selected_indices = rng.choice(total_available, size=num_to_select, replace=False)
    selected_indices.sort()

    raw_sub = raw_all[selected_indices]
    pure_sub = pure_all[selected_indices]

    print(f"Selected {num_to_select} diverse test spectra for interactive inspection.")

    # 3. Helper to resolve model paths
    def resolve_model_path(path_arg, fallback_names):
        candidates = [path_arg, os.path.join(SCRIPT_DIR, path_arg), os.path.join(PROJECT_ROOT, path_arg)]
        for f in fallback_names:
            candidates.extend([
                os.path.join(SCRIPT_DIR, f),
                os.path.join(PROJECT_ROOT, "training_round5", f),
                os.path.join(PROJECT_ROOT, "training_round4", f),
                os.path.join(PROJECT_ROOT, f)
            ])
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                return os.path.abspath(c)
        return None

    path_p2 = resolve_model_path(args.model_p2, ["head2head_polygaussnet2.pth", "polygaussnet2.pth", "polygaussnet4.pth"])
    path_p3 = resolve_model_path(args.model_p3, ["head2head_polygaussnet3.pth", "polygaussnet3.pth"])

    if path_p2 is None:
        raise FileNotFoundError(f"Could not find PolyGaussNet2 weights '{args.model_p2}'.")
    if path_p3 is None:
        raise FileNotFoundError(f"Could not find PolyGaussNet3 weights '{args.model_p3}'.")

    # 4. Load Models & Run Batch Inference (Fast precomputation)
    print("Running batch inference for PolyGaussNet2 and PolyGaussNet3...")
    m2 = PolyGaussNet2(poly_order=args.poly_order, filter_kernel_size=args.filter_kernel_size).to(device)
    m2.load_state_dict(torch.load(path_p2, map_location=device, weights_only=False))
    m2.eval()

    m3 = PolyGaussNet3(poly_order=args.poly_order, filter_kernel_size=args.filter_kernel_size, min_amplitude=0.85, max_amplitude=1.15).to(device)
    m3.load_state_dict(torch.load(path_p3, map_location=device, weights_only=False))
    m3.eval()

    res_p2 = run_model_inference(m2, raw_sub, batch_size=32, device=device)
    res_p3 = run_model_inference(m3, raw_sub, batch_size=32, device=device)

    print("✓ Inference complete. Launching interactive viewer window...")
    print("  Controls:")
    print("    • [▶ / Right / Space / d] : Next Page / Spectrum")
    print("    • [◀ / Left  / a        ] : Previous Page / Spectrum")
    print("    • [Tab / m              ] : Toggle between 4-Grid and Single Spectrum View")
    print("    • [r                    ] : Jump to Random Page / Spectrum")
    print("    • [Esc / q              ] : Exit\n")

    slideshow = Head2HeadSlideshow(raw_sub, pure_sub, res_p2, res_p3, selected_indices, per_page=args.per_page)
    slideshow.show()


if __name__ == "__main__":
    main()
