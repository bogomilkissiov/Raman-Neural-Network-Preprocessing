"""
TRAINING SCRIPT
---------------
PAUSE AND RESUME FEATURE:
- You can safely pause training at any time by pressing Ctrl+C in your terminal.
- Checkpoints are automatically saved at the end of every epoch to a file named `training_checkpoint.pth` in this directory.
- To resume, simply run `python training.py` again. It will automatically detect the checkpoint and pick up exactly where it left off, restoring the model weights, optimizer, and scheduler states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import the existing network model
import os
import glob

def load_file_data(filepath):
    """Helper function to load data from disk into PyTorch tensors."""
    data = np.load(filepath)
    X = torch.tensor(data["full_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_bc = torch.tensor(data["pure_noise_cosmic_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_clean = torch.tensor(data["pure_matrix"], dtype=torch.float32).unsqueeze(1)
    data.close()
    return X, Y_bc, Y_clean
from dual_supervised_resnet import DualSupervisedNet, LogCoshLoss

# =====================================================================
# LOSS FUNCTION WITH ASYMMETRIC PENALTY (PREVENTS PEAK HALLUCINATION)
# =====================================================================
class AsymmetricLogCoshLoss(nn.Module):
    """
    Log-Cosh Loss with an asymmetric multiplier to penalize over-predictions 
    (y_pred > y_true, i.e., hallucinating peaks/intensity) more heavily than under-predictions.
    """
    def __init__(self, overpred_penalty=2.0):
        super(AsymmetricLogCoshLoss, self).__init__()
        self.penalty = overpred_penalty

    def forward(self, y_pred, y_true):
        x = y_pred - y_true
        # Numerically stable Log-Cosh pointwise loss
        pointwise_loss = (
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device))
        )
        # Apply asymmetric penalty when prediction > ground truth
        weights = torch.where(x > 0, self.penalty, 1.0)
        return torch.mean(weights * pointwise_loss)

def train():
    # -----------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = 250
    learning_rate = 0.0004
    epochs = 140

    # Loss weighting & regularization factors
    lambda_bc = 1.5           # Increased scaling factor for latent output (baseline subtraction)
    lambda_clean = 1.0        # Scaling factor for final denoised/clean output
    lambda_l2 = 1e-4          # Tiny L2 norm penalty on pred_clean to discourage hallucinating peaks
    overpred_penalty = 2.0     # Multiplier to penalize over-predicted peaks (Option A)

    # -----------------------------------------------------------------
    # 2. FIND DATASET FILES & PREPARE
    # -----------------------------------------------------------------
    # Find all generated_spectra folders and the old "generated spectra" folder
    all_dirs = [d for d in glob.glob("generated_spectra*") if os.path.isdir(d)]
    if os.path.isdir("generated spectra"):
        all_dirs.append("generated spectra")

    if all_dirs:
        # Get the most recently modified/created dataset folder
        latest_dir = max(all_dirs, key=os.path.getmtime)
        print(f"Loading datasets from the most recent folder: {latest_dir}")
        file_list = sorted(glob.glob(os.path.join(latest_dir, "*.npz")))
    else:
        # Fallback to checking the current directory
        file_list = sorted(glob.glob("dataset_part_*.npz"))
        
    print(f"Found {len(file_list)} dataset files for training.")
    if len(file_list) == 0:
        raise FileNotFoundError("No .npz dataset files found. Please run create_dataset.py first.")

    # Determine input spectrum length from first file
    sample_data = np.load(file_list[0])
    input_length = sample_data["full_matrix"].shape[1]
    sample_data.close()

    # -----------------------------------------------------------------
    # 3. INITIALIZE MODEL, LOSS, OPTIMIZER, AND SCHEDULER
    # -----------------------------------------------------------------
    print("Initializing Model, Loss Function, Optimizer, and Scheduler...")
    model = DualSupervisedNet(input_length=input_length).to(device)
    
    criterion_bc = LogCoshLoss()
    criterion_clean = AsymmetricLogCoshLoss(overpred_penalty=overpred_penalty)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # Slower, gentler cosine decay schedule (extended T_max horizon and higher minimum LR floor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs * 1.5), eta_min=1e-5)

    # -----------------------------------------------------------------
    # 3.5 CHECKPOINT RESUME
    # -----------------------------------------------------------------
    start_epoch = 0
    checkpoint_path = "training_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at '{checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Note: Could not restore optimizer/scheduler state ({e}). Using new SGD optimizer.")
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # -----------------------------------------------------------------
    # 4. TRAINING LOOP (Iterating over files)
    # -----------------------------------------------------------------
    print("Starting training loop...")
    # Executor for asynchronous file loading
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            total_batches = 0

            # Shuffle file order each epoch for randomness
            np.random.shuffle(file_list)

            # Submit the first file load task
            future = executor.submit(load_file_data, file_list[0])

            for file_idx in range(len(file_list)):
                # Wait for the current file to finish loading
                X, Y_bc, Y_clean = future.result()
                
                # If there is a next file, start loading it asynchronously NOW
                if file_idx + 1 < len(file_list):
                    future = executor.submit(load_file_data, file_list[file_idx + 1])

                dataset = TensorDataset(X, Y_bc, Y_clean)
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    pin_memory=False  # Removed for Apple Silicon Unified Memory
                )

                for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                    batch_x = batch_x.to(device)
                    batch_y_bc = batch_y_bc.to(device)
                    batch_y_clean = batch_y_clean.to(device)

                    pred_bc, pred_clean = model(batch_x)

                    # 1. Latent baseline loss and clean reconstruction loss
                    loss_bc = criterion_bc(pred_bc, batch_y_bc)
                    loss_clean = criterion_clean(pred_clean, batch_y_clean)

                    # 2. Output L2 norm regularization to slightly penalize excess energy/peaks
                    loss_l2 = torch.mean(pred_clean ** 2)

                    # Total composite loss with weighted latent term
                    loss = (lambda_bc * loss_bc) + (lambda_clean * loss_clean) + (lambda_l2 * loss_l2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    total_batches += 1

                    # Optional: Print progress every 100 batches
                    if (batch_idx + 1) % 100 == 0:
                        print(f"Epoch [{epoch+1}/{epochs}], File [{file_idx+1}/{len(file_list)}], Step [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.6f}")

                # Deliberately clear memory before opening next file
                del X, Y_bc, Y_clean, dataset, loader

            # Average loss for the entire epoch across all files
            epoch_loss = running_loss / total_batches if total_batches > 0 else 0
            
            # Step the learning rate scheduler BEFORE reading lr
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"==> Epoch [{epoch+1}/{epochs}] completed. Average Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}")

            # Save checkpoint periodically (every epoch, overwriting to save space)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to '{checkpoint_path}'")

    except KeyboardInterrupt:
        print("\n\nTraining paused by user (Ctrl+C).")
        print(f"Progress saved. Resume by running: python training.py")

    executor.shutdown()

    # -----------------------------------------------------------------
    # 5. SAVE THE TRAINED MODEL
    # -----------------------------------------------------------------
    save_path = "dual_supervised_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to '{save_path}'")

if __name__ == "__main__":
    train()
