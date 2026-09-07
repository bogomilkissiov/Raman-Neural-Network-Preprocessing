import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet2 import PolyGaussNet
from training4 import LogCoshLoss, AsymmetricBaselinePenalty, resolve_data_path, load_file_data


def main():
    # Set seeds for reproducibility of random sample selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Load 128 spectra from the very first training file
    data_dir = resolve_data_path("training_data4")
    first_file = os.path.join(data_dir, "dataset_part_1.npz")
    
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Could not find dataset file at: {first_file}")

    X, Y_bc, Y_clean = load_file_data(first_file)
    
    num_samples = 8
    batch_x = X[:num_samples].to(device)
    batch_y_bc = Y_bc[:num_samples].to(device)
    batch_y_clean = Y_clean[:num_samples].to(device)

    # 2. Initialize Model and Current Training Regime (Round 4)
    poly_order = 7
    filter_kernel_size = 31
    lr = 9e-4
    weight_decay = 0
    epochs = 5000
    lambda_bc = 1.0
    lambda_clean = 1.0
    lambda_asym = 10.0

    model = PolyGaussNet(poly_order=poly_order, filter_kernel_size=filter_kernel_size).to(device)
    criterion = LogCoshLoss().to(device)
    asym_penalty = AsymmetricBaselinePenalty(lambda_asym=lambda_asym).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Train on the 16 spectra
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        clean_pred, pred_baseline, bc_pred, _ = model(batch_x)
        
        loss_bc = criterion(bc_pred, batch_y_bc)
        loss_clean = criterion(clean_pred, batch_y_clean)
        loss_asym = asym_penalty(pred_baseline, batch_x)
        
        total_loss = (lambda_bc * loss_bc) + (lambda_clean * loss_clean) + loss_asym
        total_loss.backward()
        optimizer.step()

    # 4. Inference on the trained spectra
    model.eval()
    with torch.no_grad():
        clean_preds, pred_baselines, _, _ = model(batch_x)

    # Convert to CPU numpy
    raw_np = batch_x.squeeze(1).cpu().numpy()
    pure_np = batch_y_clean.squeeze(1).cpu().numpy()
    baseline_np = pred_baselines.squeeze(1).cpu().numpy()
    clean_np = clean_preds.squeeze(1).cpu().numpy()

    # 5. Plot the same 3 spectra in the exact style of the reference figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=150)
    
    for i in range(3):
        ax = axes[i]
        ax.plot(raw_np[i], label="Raw Noisy", color="#b0b0b0", lw=1.2, alpha=0.9)
        ax.plot(baseline_np[i], label=f"Pred Baseline (Poly{poly_order})", color="orange", linestyle="--", lw=1.5)
        ax.plot(clean_np[i], label="Pred Clean (PolyGaussNet)", color="red", lw=1.3)
        ax.plot(pure_np[i], label="Ground Truth Pure", color="black", lw=1.2)

        ax.set_title(f"Sample {i + 1}", fontsize=12, pad=6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.tick_params(direction="out")

    plt.tight_layout()

    # Save output plot
    save_path = os.path.join(SCRIPT_DIR, "overfit_check.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
