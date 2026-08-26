"""
EVALUATION ON EXPERIMENTAL RAMAN MAP (2s_100lp_map-2)
------------------------------------------------------
Processes real experimental Raman map data using both:
1. Conventional Preprocessing Pipeline (pre.py)
2. PolyGaussNet3 (polygaussnet3.pth)

Generates 3 publication-quality multi-stage breakdown plots styled in the
Gruvbox Dark aesthetic (matching the Residual Breakdown design).
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from renishawWiRE import WDFReader

# Configure imports so they resolve whether running from project root or inside training_round3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pre
from spectra_class import spectra
from polygaussnet import PolyGaussNet
import gruvbox_theme
from gruvbox_theme import GRUVBOX


def apply_gruvbox_styling():
    """Configures matplotlib rcParams for Gruvbox dark aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": GRUVBOX.get("bg0", "#282828"),
        "axes.facecolor": GRUVBOX.get("bg0_hard", "#1d2021"),
        "axes.edgecolor": GRUVBOX.get("bg3", "#665c54"),
        "axes.linewidth": 1.0,
        "axes.labelcolor": GRUVBOX.get("fg", "#ebdbb2"),
        "axes.titlecolor": GRUVBOX.get("fg0", "#fbf1c7"),
        "xtick.color": GRUVBOX.get("fg4", "#a89984"),
        "ytick.color": GRUVBOX.get("fg4", "#a89984"),
        "grid.color": GRUVBOX.get("bg2", "#504945"),
        "grid.alpha": 0.45,
        "grid.linestyle": ":",
        "text.color": GRUVBOX.get("fg", "#ebdbb2"),
        "legend.facecolor": GRUVBOX.get("bg0_soft", "#32302f"),
        "legend.edgecolor": GRUVBOX.get("bg3", "#665c54"),
        "legend.labelcolor": GRUVBOX.get("fg1", "#ebdbb2"),
        "legend.framealpha": 0.92,
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    })


def main():
    print("=" * 78)
    print(" Processing Real Experimental Dataset: 2s_100lp_map-2 (1).wdf")
    print("=" * 78)

    possible_wdf_paths = [
        "2s_100lp_map-2 (1).wdf",
        os.path.join(SCRIPT_DIR, "2s_100lp_map-2 (1).wdf"),
        os.path.join(PROJECT_ROOT, "2s_100lp_map-2 (1).wdf"),
    ]
    wdf_filename = next((p for p in possible_wdf_paths if os.path.exists(p)), None)
    if wdf_filename is None:
        raise FileNotFoundError("Cannot find WDF file '2s_100lp_map-2 (1).wdf'")

    # 1. Load WDF dataset
    print(f"Loading '{wdf_filename}' with WDFReader...")
    wdf = WDFReader(wdf_filename)
    wn = wdf.xdata.astype(np.float32)

    # Ensure wavenumbers are sorted ascending
    if wn[0] > wn[-1]:
        sort_idx = np.argsort(wn)
        wn = wn[sort_idx]
        all_raw = wdf.spectra.reshape(-1, len(wn))[:, sort_idx].astype(np.float32)
    else:
        all_raw = wdf.spectra.reshape(-1, len(wn)).astype(np.float32)

    total_spectra = len(all_raw)
    print(f"Loaded {total_spectra:,} spectra | Wavenumbers: {wn[0]:.1f} to {wn[-1]:.1f} cm⁻¹ ({len(wn)} bins)\n")

    # 2. Select 3 representative spectra across the map (e.g. low, medium, high intensity)
    selected_indices = [500, 2500, 7500]
    raw_selected = all_raw[selected_indices]
    wn_selected = np.tile(wn, (len(selected_indices), 1))

    # 3. Process with Conventional Pipeline (pre.py)
    print("Running Conventional Preprocessing Pipeline (Despike + Wavelet + Baseline)...")
    conv_data = spectra.from_matrices(wn_selected, np.copy(raw_selected))
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    conv_clean = conv_data.intensity_matrix

    # 4. Process with PolyGaussNet3
    # Note: PolyGaussNet was trained on spectra shifted so min_value = 0 and scaled to training domain (mean ~ 0.892).
    # We shift min to 0, scale by mean / 0.892, run inference, and then reconstruct the physical count units.
    print("Running PolyGaussNet3 (Model: polygaussnet3.pth)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PolyGaussNet().to(device)

    possible_model_paths = [
        "polygaussnet3.pth",
        "test_polygaussnet3.pth",
        os.path.join(SCRIPT_DIR, "polygaussnet3.pth"),
        os.path.join(SCRIPT_DIR, "test_polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "test_polygaussnet3.pth"),
    ]
    model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError("Could not find polygaussnet3.pth or test_polygaussnet3.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Step A: Shift min to 0
    mins = np.min(raw_selected, axis=1, keepdims=True)
    raw_shifted = raw_selected - mins

    # Step B: Scale per-spectrum to match training distribution (~0.892 mean)
    scales = np.mean(raw_shifted, axis=1, keepdims=True) / 0.892
    raw_scaled = raw_shifted / scales

    with torch.no_grad():
        raw_tensor = torch.tensor(raw_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        poly_clean_t, poly_base_t, poly_bc_t, _ = model(raw_tensor)

    # Step C: Scale and un-shift back to experimental physical counts
    poly_clean = poly_clean_t.squeeze(1).cpu().numpy() * scales
    poly_base = poly_base_t.squeeze(1).cpu().numpy() * scales + mins
    poly_bc = poly_bc_t.squeeze(1).cpu().numpy() * scales

    # 5. Generate 3 unique Gruvbox multi-stage comparison plots
    print("\nGenerating publication-quality Gruvbox plots for the 3 spectra...")
    apply_gruvbox_styling()

    COLOR_RAW = GRUVBOX.get("gray", "#928374")
    COLOR_CONV = GRUVBOX.get("orange", "#fe8019")
    COLOR_POLY = GRUVBOX.get("green", "#b8bb26")
    COLOR_BASE = GRUVBOX.get("yellow", "#fabd2f")
    COLOR_LATENT = GRUVBOX.get("purple", "#d3869b")
    COLOR_BG = GRUVBOX.get("bg0_hard", "#1d2021")
    COLOR_FG = GRUVBOX.get("fg0", "#fbf1c7")

    output_filenames = []

    for i, orig_idx in enumerate(selected_indices):
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, dpi=180, facecolor=GRUVBOX.get("bg0", "#282828"))

        for ax in axes:
            ax.set_facecolor(COLOR_BG)

        # Stage 1: Raw Spectrum + Inferred Polynomial Baseline
        axes[0].plot(wn, raw_selected[i], color=COLOR_RAW, label="Raw Experimental Spectrum", alpha=0.7, linewidth=0.85)
        axes[0].plot(wn, poly_base[i], color=COLOR_BASE, label="PolyGaussNet Inferred Polynomial Baseline (Order 15)", linewidth=1.2)
        axes[0].set_title(f"Stage 1: Polynomial Baseline Subtraction (Spectrum #{orig_idx + 1})", fontsize=11, fontweight="bold", color=COLOR_FG)
        axes[0].set_ylabel("Intensity (Counts)", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[0].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[0].grid(True, linestyle=":", alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 2: Baseline-Corrected Signal
        axes[1].plot(wn, poly_bc[i], color=COLOR_LATENT, label="PolyGaussNet Baseline-Corrected (Raw − Baseline)", alpha=0.9, linewidth=0.9)
        axes[1].set_title(f"Stage 2: Latent Baseline-Corrected Raman Bands", fontsize=11, fontweight="bold", color=COLOR_FG)
        axes[1].set_ylabel("Intensity (Counts)", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[1].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[1].grid(True, linestyle=":", alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 3: Clean Processed Spectra (Conventional vs PolyGaussNet3 Overlap)
        axes[2].plot(wn, conv_clean[i], color=COLOR_CONV, label="Conventional Pipeline (Despike + Wavelet + Baseline)", alpha=0.8, linewidth=0.9)
        axes[2].plot(wn, poly_clean[i], color=COLOR_POLY, label="PolyGaussNet3 Inferred Output (Adaptive GenGaussian Filter)", linewidth=1.1)
        axes[2].set_title(f"Stage 3: Denoised & Baseline-Free Spectrum Comparison (Conventional vs PolyGaussNet3)", fontsize=11, fontweight="bold", color=COLOR_FG)
        axes[2].set_xlabel("Raman Shift / Wavenumber (cm⁻¹)", fontsize=10, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[2].set_ylabel("Intensity (Counts)", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[2].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[2].grid(True, linestyle=":", alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        plt.suptitle(
            f"Experimental Raman Map Preprocessing Comparison | Spectrum #{orig_idx + 1} (File: {wdf_filename})",
            fontsize=12,
            fontweight="bold",
            color=COLOR_FG,
            y=0.99
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        out_name = f"real_spectrum_comparison_{i+1}_idx_{orig_idx+1}.png"
        plt.savefig(out_name, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

        output_filenames.append(out_name)
        print(f"  ✓ Saved plot {i+1} to: '{out_name}'")

    print("\n" + "=" * 78)
    print(" All 3 real spectrum comparison plots successfully generated!")
    print("=" * 78)


if __name__ == "__main__":
    main()
