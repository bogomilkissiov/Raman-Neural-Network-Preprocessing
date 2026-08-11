import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Import the existing network model
import os
import glob
from dual_supervised_resnet import DualSupervisedNet, LogCoshLoss

def train():
    # -----------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = 48
    learning_rate = 0.001
    epochs = 10

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
    # 3. INITIALIZE MODEL, LOSS, AND SGD OPTIMIZER
    # -----------------------------------------------------------------
    print("Initializing Model, Loss Function, and Optimizer...")
    model = DualSupervisedNet(input_length=input_length).to(device)
    
    criterion = LogCoshLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # -----------------------------------------------------------------
    # 4. TRAINING LOOP (Iterating over files)
    # -----------------------------------------------------------------
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        # Shuffle file order each epoch for randomness
        np.random.shuffle(file_list)

        for file_idx, filepath in enumerate(file_list):
            # Load ONE file into memory at a time
            data = np.load(filepath)
            
            X = torch.tensor(data["full_matrix"], dtype=torch.float32).unsqueeze(1)
            Y_bc = torch.tensor(data["pure_noise_cosmic_matrix"], dtype=torch.float32).unsqueeze(1)
            Y_clean = torch.tensor(data["pure_matrix"], dtype=torch.float32).unsqueeze(1)
            
            data.close()

            dataset = TensorDataset(X, Y_bc, Y_clean)
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=True if device.type != 'cpu' else False
            )

            for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                batch_x = batch_x.to(device)
                batch_y_bc = batch_y_bc.to(device)
                batch_y_clean = batch_y_clean.to(device)

                pred_bc, pred_clean = model(batch_x)

                loss_bc = criterion(pred_bc, batch_y_bc)
                loss_clean = criterion(pred_clean, batch_y_clean)
                loss = loss_bc + loss_clean

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
        print(f"==> Epoch [{epoch+1}/{epochs}] completed. Average Loss: {epoch_loss:.6f}")

    # -----------------------------------------------------------------
    # 5. SAVE THE TRAINED MODEL
    # -----------------------------------------------------------------
    save_path = "dual_supervised_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to '{save_path}'")

if __name__ == "__main__":
    train()
