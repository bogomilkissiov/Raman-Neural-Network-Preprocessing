import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import math
import numpy as np
import os

# Import the existing generator and network model
from spectra_generator import generate_spectra
from dual_supervised_resnet import DualSupervisedNet, LogCoshLoss

# -----------------------------------------------------------------
# ON-THE-FLY SYNTHETIC DATASET
# -----------------------------------------------------------------
class OnTheFlyRamanDataset(IterableDataset):
    """
    Generates batches of synthetic Raman spectra dynamically on-the-fly.
    """
    def __init__(self, samples_per_epoch, batch_size=256, spectra_kwargs=None):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.spectra_kwargs = spectra_kwargs or {
            "wavenum_range": [0, 1200],
            "num_peaks_range": [10, 100],
            "amplitude_range": [0.009, 0.1],
            "width_range": [2, 30],
            "degree_range": [1, 7],
            "offset_range": [0.0, 2.0],
            "max_coeff": 1.0,
            "min_peak_ratio": 2.5,
            "std_range": [0.3, 0.7],
            "probability_cosmic": 0.00001,
            "intensity_range_cosmic": [3.0, 10.0]
        }

    def __iter__(self):
        num_batches = int(math.ceil(self.samples_per_epoch / self.batch_size))

        for _ in range(num_batches):
            pure_matrix, pure_noise_cosmic_matrix, full_matrix = generate_spectra(
                batch_size=self.batch_size,
                **self.spectra_kwargs
            )
            
            # Convert numpy arrays to float32 Tensors with shape (B, 1, L)
            X = torch.tensor(full_matrix, dtype=torch.float32).unsqueeze(1)
            Y_bc = torch.tensor(pure_noise_cosmic_matrix, dtype=torch.float32).unsqueeze(1)
            Y_clean = torch.tensor(pure_matrix, dtype=torch.float32).unsqueeze(1)
            
            # Yield individual samples to the DataLoader
            for i in range(self.batch_size):
                yield X[i], Y_bc[i], Y_clean[i]

def train():
    # -----------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters for overnight training
    batch_size = 256
    samples_per_epoch = 240000 
    epochs = 120
    learning_rate = 0.0004
    loss_log_file = "epoch_losses.txt"

    # Initialize the loss log file with headers
    with open(loss_log_file, "w") as f:
        f.write("Epoch\tAverage_Loss\n")

    # -----------------------------------------------------------------
    # 2. CREATE ON-THE-FLY DATASET & DATALOADER
    # -----------------------------------------------------------------
    print(f"Setting up on-the-fly data generation (Batch Size: {batch_size}, Samples/Epoch: {samples_per_epoch})...")
    dataset = OnTheFlyRamanDataset(samples_per_epoch=samples_per_epoch, batch_size=batch_size)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=True if device.type != 'cpu' else False
    )

    # -----------------------------------------------------------------
    # 3. INITIALIZE MODEL, LOSS, AND OPTIMIZER
    # -----------------------------------------------------------------
    print("Initializing Model, Loss Function, and Optimizer...")
    
    # Generate one small batch to determine the sequence length dynamically
    sample_x, _, _ = next(iter(dataset))
    input_length = sample_x.shape[-1]
    
    model = DualSupervisedNet(input_length=input_length).to(device)
    criterion = LogCoshLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # -----------------------------------------------------------------
    # 4. TRAINING LOOP
    # -----------------------------------------------------------------
    print(f"Starting overnight training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # We manually keep track of the number of batches processed to compute the average loss
        num_batches_processed = 0

        for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
            # Move tensors to the target device
            batch_x = batch_x.to(device)
            batch_y_bc = batch_y_bc.to(device)
            batch_y_clean = batch_y_clean.to(device)

            # Forward pass
            pred_bc, pred_clean = model(batch_x)

            # Sum of LogCosh losses for both supervision targets
            loss = criterion(pred_bc, batch_y_bc) + criterion(pred_clean, batch_y_clean)

            # Zero gradients, backward pass, and SGD optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches_processed += 1

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}], Loss: {loss.item():.6f}")

        # Average loss for the entire epoch
        epoch_loss = running_loss / num_batches_processed
        print(f"==> Epoch [{epoch+1}/{epochs}] completed. Average Loss: {epoch_loss:.6f}")

        # Save loss to txt file
        with open(loss_log_file, "a") as f:
            f.write(f"{epoch+1}\t{epoch_loss:.6f}\n")

        # Save model checkpoint
        latest_checkpoint_path = f"dual_supervised_resnet_latest.pth"
        torch.save(model.state_dict(), latest_checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            interval_checkpoint_path = f"dual_supervised_resnet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), interval_checkpoint_path)
            print(f"Interval checkpoint saved at '{interval_checkpoint_path}'")

    print("\nTraining complete! Final report:")
    print("--------------------------------")
    with open(loss_log_file, "r") as f:
        print(f.read())
    
    print("Model weights have been saved.")

if __name__ == "__main__":
    train()
