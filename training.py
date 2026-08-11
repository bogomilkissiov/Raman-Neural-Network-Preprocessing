import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import the existing generator and network model
from spectra_generator import generate_spectra
from dual_supervised_resnet import DualSupervisedNet, LogCoshLoss

def train():
    # -----------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    num_samples = 64
    batch_size = 16
    learning_rate = 0.01
    epochs = 100

    # -----------------------------------------------------------------
    # 2. GENERATE SYNTHETIC DATA & PREPARE DATALOADER
    # -----------------------------------------------------------------
    print(f"Generating {num_samples} synthetic spectra... This may take a moment.")
    pure_matrix, pure_noise_cosmic_matrix, full_matrix = generate_spectra(
        batch_size=num_samples,
        wavenum_range=[0, 1200],          
        num_peaks_range=[5, 30],
        amplitude_range=[0.1, 1.0],
        width_range=[2, 30],
        degree_range=[1, 7],
        offset_range=[0.0, 0.5],
        max_coeff=1.0,
        min_peak_ratio=2,
        std_range=[0.01, 0.05],
        probability_cosmic=0.0001,
        intensity_range_cosmic=[3.0, 7.0]
    )

    print("Data generation complete. Converting to PyTorch Tensors...")
    
    # Convert numpy matrices directly into 3D PyTorch tensors: Shape (N, 1, L)
    X = torch.tensor(full_matrix, dtype=torch.float32).unsqueeze(1)
    Y_bc = torch.tensor(pure_noise_cosmic_matrix, dtype=torch.float32).unsqueeze(1)
    Y_clean = torch.tensor(pure_matrix, dtype=torch.float32).unsqueeze(1)

    # Wrap into PyTorch's native TensorDataset and DataLoader
    dataset = TensorDataset(X, Y_bc, Y_clean)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if device.type != 'cpu' else 0,  # Use workers if on GPU/MPS
        pin_memory=True if device.type != 'cpu' else False
    )

    # -----------------------------------------------------------------
    # 3. INITIALIZE MODEL, LOSS, AND SGD OPTIMIZER
    # -----------------------------------------------------------------
    print("Initializing Model, Loss Function, and Optimizer...")
    input_length = X.shape[-1]
    model = DualSupervisedNet(input_length=input_length).to(device)
    
    criterion = LogCoshLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # -----------------------------------------------------------------
    # 4. TRAINING LOOP
    # -----------------------------------------------------------------
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
            # Move tensors to the target device
            batch_x = batch_x.to(device)
            batch_y_bc = batch_y_bc.to(device)
            batch_y_clean = batch_y_clean.to(device)

            # Forward pass: get intermediate (baseline removal) and final (clean) predictions
            pred_bc, pred_clean = model(batch_x)

            # Simple sum of LogCosh losses for both supervision targets
            loss_bc = criterion(pred_bc, batch_y_bc)
            loss_clean = criterion(pred_clean, batch_y_clean)
            loss = loss_bc + loss_clean

            # Zero gradients, backward pass, and SGD optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Optional: Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.6f}")

        # Average loss for the entire epoch
        epoch_loss = running_loss / len(loader)
        print(f"==> Epoch [{epoch+1}/{epochs}] completed. Average Loss: {epoch_loss:.6f}")

    # -----------------------------------------------------------------
    # 5. SAVE THE TRAINED MODEL
    # -----------------------------------------------------------------
    save_path = "dual_supervised_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to '{save_path}'")

if __name__ == "__main__":
    train()
