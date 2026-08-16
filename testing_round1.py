import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

from spectra_class import spectrum, spectra
import preprocessing as pp
from dual_supervised_resnet import DualSupervisedNet
import file_loaders as fl

# -----------------------------------------------------------------------------
# 1. LOAD TEST DATASET FILES
# -----------------------------------------------------------------------------
# Sort files naturally by part number
test_files = sorted(
    glob.glob(os.path.join("test_spectra", "dataset_part_*.npz")),
    key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])
)[:5]

print(f"Loading {len(test_files)} dataset files:\n")
for f in test_files:
    print(f" - {f}")

pure_list = []
pure_noise_cosmic_list = []
full_list = []

for filepath in test_files:
    with np.load(filepath) as data:
        pure_list.append(data["pure_matrix"])
        pure_noise_cosmic_list.append(data["pure_noise_cosmic_matrix"])
        full_list.append(data["full_matrix"])

# Stack all 5 parts vertically into large continuous 2D matrices
pure_intensities = np.vstack(pure_list)
pure_noise_cosmic_intensities = np.vstack(pure_noise_cosmic_list)
composite_intensities = np.vstack(full_list)

# Combined into one large tuple
test_data = (pure_intensities, pure_noise_cosmic_intensities, composite_intensities)

# -----------------------------------------------------------------------------
# 2. INITIALIZE SPECTRA OBJECTS
# -----------------------------------------------------------------------------
# Generate wavenumber grid matching the spectral length (0 to 1200 cm^-1, step size 1)
num_wavenumbers = test_data[0].shape[1]
wavenumbers = np.arange(0, num_wavenumbers, 1)

# Broadcast wavenumbers to 2D matrix matching the number of spectra
wavenumber_matrix = np.tile(wavenumbers, (test_data[0].shape[0], 1))

pure_spectra = spectra.from_matrices(wavenumber_matrix, test_data[0])
noisy_spectra = spectra.from_matrices(wavenumber_matrix, test_data[1])

raw_spectra = spectra.from_matrices(wavenumber_matrix, test_data[2])

print("Spectra objects initialized.\n")

# -----------------------------------------------------------------------------
# 3. INITIALIZE PYTORCH TENSORS (Shape: N, 1, L)
# -----------------------------------------------------------------------------
pure_tensor = torch.tensor(test_data[0], dtype=torch.float32).unsqueeze(1)
noisy_tensor = torch.tensor(test_data[1], dtype=torch.float32).unsqueeze(1)

raw_tensor = torch.tensor(test_data[2], dtype=torch.float32).unsqueeze(1)

print("PyTorch tensors initialized.\n")
# -----------------------------------------------------------------------------
# 4. APPLY PREPROCESSING PIPELINE TO RAW SPECTRA
# -----------------------------------------------------------------------------
print("Applying preprocessing pipeline...\n")
processed_spectra = pp.preprocess_pipeline(raw_spectra, normalize=False, shift=False)

# -----------------------------------------------------------------------------
# 5. RUN RAW TENSORS THROUGH TRAINED NEURAL NETWORK
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading neural network...\n")

# 1. Instantiate the architecture
model = DualSupervisedNet(input_length=num_wavenumbers).to(device)

# 2. Load the trained model weights
model_weights_path = "dual_supervised_resnet.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
print(f"Neural network loaded successfully.\n")

# 3. Set to evaluation mode
model.eval()

print(f"Running neural network inference on device...\n")
# 4. Forward pass
with torch.no_grad():
    inputs = raw_tensor.to(device)
    latent_bc_tensor, processed_tensor = model(inputs)
    processed_tensor = processed_tensor.cpu()
    latent_bc_tensor = latent_bc_tensor.cpu()

# -----------------------------------------------------------------------------
# 6. CONVERT PREDICTIONS TO NUMPY ARRAYS
# -----------------------------------------------------------------------------
# Squeeze out channel dimension (N, 1, L) -> (N, L)
processed_matrix = processed_tensor.squeeze(1).numpy()
latent_bc_matrix = latent_bc_tensor.squeeze(1).numpy()
print("Neural network inference completed.\n")

# -----------------------------------------------------------------------------
# 7. CALCULATE AVERAGE COSINE, MSE, LOG COSH
# -----------------------------------------------------------------------------
print("Calculating metrics...")
def calculate_metrics(y_true, y_pred):
    """
    Computes spectral metrics between ground truth and predicted matrices.
    y_true, y_pred shape: (N, L)
    """
    # 1. Cosine Similarity per spectrum
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1) + 1e-12
    norm_pred = np.linalg.norm(y_pred, axis=1) + 1e-12
    cosine_sim = dot_product / (norm_true * norm_pred)
    
    # 2. Mean Squared Error (MSE) per spectrum
    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    
    # 3. Log-Cosh Loss per spectrum (numerically stable: |x| + log1p(exp(-2|x|)) - log(2))
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    log_cosh_elementwise = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0)
    log_cosh = np.mean(log_cosh_elementwise, axis=1)
    
    return {
        "Cosine Similarity": (np.mean(cosine_sim), np.std(cosine_sim)),
        "MSE": (np.mean(mse), np.std(mse)),
        "Log-Cosh": (np.mean(log_cosh), np.std(log_cosh)),
    }

ground_truth = pure_spectra.intensity_matrix
conv_pred = processed_spectra.intensity_matrix
nn_pred = processed_matrix

metrics_conv = calculate_metrics(ground_truth, conv_pred)
metrics_nn = calculate_metrics(ground_truth, nn_pred)

print("\n" + "=" * 65)
print(f"{'EVALUATION RESULTS vs PURE SPECTRA (N = ' + str(ground_truth.shape[0]) + ')':^65}")
print("=" * 65)
print(f"{'Metric':<22} | {'Conventional Pipeline':<18} | {'Neural Network':<18}")
print("-" * 65)
for metric in ["Cosine Similarity", "MSE", "Log-Cosh"]:
    conv_val, conv_std = metrics_conv[metric]
    nn_val, nn_std = metrics_nn[metric]
    print(f"{metric:<22} | {conv_val:10.6f} (±{conv_std:.4f}) | {nn_val:10.6f} (±{nn_std:.4f})")
print("=" * 65 + "\n")

# -----------------------------------------------------------------------------
# 8. PLOTTING INDIVIDUAL SPECTRA
# -----------------------------------------------------------------------------
print("Launching interactive spectra viewer...")
import gruvbox_theme
from gruvbox_theme import GRUVBOX

class SpectrumSlideshow:
    """
    Interactive Matplotlib viewer for inspecting Raman spectra.
    Shows 4 stitched subplots in Gruvbox style:
      - Ground Truth (Pure)
      - Raw Unprocessed (Composite)
      - Conventional Preprocessed
      - Neural Network Preprocessed
    """
    def __init__(self, wavenumbers, pure_mat, raw_mat, conv_mat, nn_mat, num_samples=100):
        self.wavenumbers = wavenumbers
        
        # Pick 100 samples from the test set
        total_available = len(pure_mat)
        self.num_samples = min(num_samples, total_available)
        
        # Select first num_samples (or random sample if preferred)
        self.indices = np.arange(self.num_samples)
        self.pure = pure_mat[self.indices]
        self.raw = raw_mat[self.indices]
        self.conv = conv_mat[self.indices]
        self.nn = nn_mat[self.indices]
        
        self.current_idx = 0
        
        # Create 2x2 grid of subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120, sharex=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial draw
        self.update_plot()
        plt.show()

    def update_plot(self):
        idx = self.current_idx
        sample_num = self.indices[idx]
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
            
        ax_pure = self.axes[0, 0]
        ax_raw = self.axes[0, 1]
        ax_conv = self.axes[1, 0]
        ax_nn = self.axes[1, 1]
        
        # 1. Ground Truth (Pure)
        ax_pure.plot(self.wavenumbers, self.pure[idx], color=GRUVBOX["blue"], linewidth=1.8, label="Pure Spectrum")
        ax_pure.set_title("Ground Truth (Pure)", fontsize=11, fontweight="bold")
        ax_pure.set_ylabel("Intensity")
        ax_pure.legend(loc="upper right")
        
        # 2. Raw Unprocessed (Composite)
        ax_raw.plot(self.wavenumbers, self.raw[idx], color=GRUVBOX["gray"], linewidth=1.5, label="Raw Composite")
        ax_raw.set_title("Raw Unprocessed (Peaks + Baseline + Noise + Cosmic)", fontsize=11, fontweight="bold")
        ax_raw.legend(loc="upper right")
        
        # 3. Conventional Pipeline
        ax_conv.plot(self.wavenumbers, self.conv[idx], color=GRUVBOX["orange"], linewidth=1.8, label="Conventional Pipeline")
        ax_conv.set_title("Conventional Preprocessed (Despike + Denoise + Baseline)", fontsize=11, fontweight="bold")
        ax_conv.set_xlabel("Wavenumber (cm⁻¹)")
        ax_conv.set_ylabel("Intensity")
        ax_conv.legend(loc="upper right")
        
        # 4. Neural Network
        ax_nn.plot(self.wavenumbers, self.nn[idx], color=GRUVBOX["green"], linewidth=1.8, label="Neural Network")
        ax_nn.set_title("Neural Network Preprocessed (Dual-Supervised ResNet)", fontsize=11, fontweight="bold")
        ax_nn.set_xlabel("Wavenumber (cm⁻¹)")
        ax_nn.legend(loc="upper right")
        
        # Overall figure super-title
        self.fig.suptitle(
            f"Spectrum Sample {idx + 1} / {self.num_samples} (Test Set #{sample_num})  |  Use [◀ / ▶] Arrow Keys to Navigate (ESC to Close)",
            fontsize=13,
            fontweight="bold",
            color=GRUVBOX["fg0"],
            y=0.98
        )
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', ' ', 'down']:
            self.current_idx = (self.current_idx + 1) % self.num_samples
            self.update_plot()
        elif event.key in ['left', 'up']:
            self.current_idx = (self.current_idx - 1) % self.num_samples
            self.update_plot()
        elif event.key in ['escape', 'q']:
            plt.close(self.fig)

# Launch the interactive viewer for 100 spectra
SpectrumSlideshow(
    wavenumbers=wavenumbers,
    pure_mat=ground_truth,
    raw_mat=test_data[2],
    conv_mat=conv_pred,
    nn_mat=nn_pred,
    num_samples=20)

# -----------------------------------------------------------------------------
# 9. PRELIMINARY RUN ON REAL DATA
# -----------------------------------------------------------------------------
real_raw_data_full = fl.wdf_to_spectra("2s_100lp_map-2 (1).wdf")

# Slice out only the first 20 spectra to save memory and speed up computation
real_raw_data = spectra.from_matrices(
    real_raw_data_full.wavenumber_matrix[:20],
    real_raw_data_full.intensity_matrix[:20],
    real_raw_data_full.position_matrix[:20] if real_raw_data_full.position_matrix is not None else None
)

# only normalize and shift so NN can process it
real_raw_spectra = pp.preprocess_pipeline(copy.deepcopy(real_raw_data), despike=False, denoise=False, baseline=False)

# Convert to tensor: shape (20, 1, 1015)
real_raw_tensor_orig = torch.tensor(real_raw_spectra.intensity_matrix, dtype=torch.float32).unsqueeze(1)

# Resample / stretch intensity matrix to exactly 1201 bins expected by the neural network
import torch.nn.functional as F
real_raw_tensor = F.interpolate(real_raw_tensor_orig, size=num_wavenumbers, mode='linear', align_corners=True)

with torch.no_grad():
    inputs = real_raw_tensor.to(device)
    real_latent_bc_tensor, real_processed_tensor = model(inputs)
    real_processed_tensor = real_processed_tensor.cpu()
    real_latent_bc_tensor = real_latent_bc_tensor.cpu()

real_processed_matrix = real_processed_tensor.squeeze(1).numpy()
real_latent_bc_matrix = real_latent_bc_tensor.squeeze(1).numpy()

real_processed_spectra = pp.preprocess_pipeline(copy.deepcopy(real_raw_data), despike=True, denoise=True, baseline=True, normalize=False, shift=True)

# Interpolate real wavenumbers to 1201 points to match the resampled NN spectra
orig_wn = real_raw_data.wavenumber_matrix[0]
resampled_wn = np.linspace(orig_wn[0], orig_wn[-1], num_wavenumbers)

# Convert resampled NN input tensor to numpy matrix
real_nn_input_matrix = real_raw_tensor.squeeze(1).numpy()

# -----------------------------------------------------------------------------
# 10. REAL DATA INTERACTIVE SLIDESHOW
# -----------------------------------------------------------------------------
class RealSpectrumSlideshow:
    """
    Interactive Matplotlib viewer for real experimental Raman data.
    Shows 4 stitched subplots in Gruvbox style:
      - Raw Unprocessed Spectrum (Fed into Neural Network: Normalized & Resampled)
      - Neural Network Latent Stage (Baseline Corrected)
      - Conventional Preprocessed (Despike + Denoise + Baseline + Shift)
      - Neural Network Fully Preprocessed (Dual-Supervised ResNet)
    """
    def __init__(self, orig_wavenumbers, resampled_wavenumbers, nn_input_mat, latent_bc_mat, conv_mat, nn_mat, num_samples=20):
        self.orig_wavenumbers = orig_wavenumbers
        self.resampled_wavenumbers = resampled_wavenumbers
        total_available = len(nn_input_mat)
        self.num_samples = min(num_samples, total_available)
        
        self.indices = np.arange(self.num_samples)
        self.nn_input = nn_input_mat[self.indices]
        self.latent_bc = latent_bc_mat[self.indices]
        self.conv = conv_mat[self.indices]
        self.nn = nn_mat[self.indices]
        
        self.current_idx = 0
        
        # Create 2x2 grid of subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_plot()
        plt.show()

    def update_plot(self):
        idx = self.current_idx
        sample_num = self.indices[idx]
        
        for ax in self.axes.flat:
            ax.clear()
            
        ax_raw = self.axes[0, 0]
        ax_latent = self.axes[0, 1]
        ax_conv = self.axes[1, 0]
        ax_nn = self.axes[1, 1]
        
        # 1. Exact Input to Neural Network (Normalized, Shifted & Resampled to 1201 pts)
        ax_raw.plot(self.resampled_wavenumbers, self.nn_input[idx], color=GRUVBOX["gray"], linewidth=1.5, label="NN Raw Input (Scaled/Shifted)")
        ax_raw.set_title("Input to Neural Network (Normalized & Shifted)", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("Wavenumber (cm⁻¹)")
        ax_raw.set_ylabel("Intensity")
        ax_raw.legend(loc="upper right")
        
        # 2. Neural Network Latent Baseline Corrected (resampled resolution)
        ax_latent.plot(self.resampled_wavenumbers, self.latent_bc[idx], color=GRUVBOX["yellow"], linewidth=1.8, label="NN Intermediate (Baseline Removed)")
        ax_latent.set_title("Neural Network: Latent Baseline Corrected", fontsize=11, fontweight="bold")
        ax_latent.set_xlabel("Wavenumber (cm⁻¹)")
        ax_latent.legend(loc="upper right")
        
        # 3. Conventional Pipeline Preprocessed (original resolution)
        ax_conv.plot(self.orig_wavenumbers, self.conv[idx], color=GRUVBOX["orange"], linewidth=1.8, label="Conventional Pipeline")
        ax_conv.set_title("Conventional Preprocessed (Despike + Denoise + Baseline)", fontsize=11, fontweight="bold")
        ax_conv.set_xlabel("Wavenumber (cm⁻¹)")
        ax_conv.set_ylabel("Intensity")
        ax_conv.legend(loc="upper right")
        
        # 4. Neural Network Final Clean (resampled resolution)
        ax_nn.plot(self.resampled_wavenumbers, self.nn[idx], color=GRUVBOX["green"], linewidth=1.8, label="Neural Network Clean")
        ax_nn.set_title("Neural Network: Fully Preprocessed", fontsize=11, fontweight="bold")
        ax_nn.set_xlabel("Wavenumber (cm⁻¹)")
        ax_nn.legend(loc="upper right")
        
        # Overall title
        self.fig.suptitle(
            f"Real Data Spectrum {idx + 1} / {self.num_samples} (Map Index #{sample_num})  |  Use [◀ / ▶] Arrow Keys to Navigate (ESC to Close)",
            fontsize=13,
            fontweight="bold",
            color=GRUVBOX["fg0"],
            y=0.98
        )
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', ' ', 'down']:
            self.current_idx = (self.current_idx + 1) % self.num_samples
            self.update_plot()
        elif event.key in ['left', 'up']:
            self.current_idx = (self.current_idx - 1) % self.num_samples
            self.update_plot()
        elif event.key in ['escape', 'q']:
            plt.close(self.fig)

# Launch the real data slideshow
RealSpectrumSlideshow(
    orig_wavenumbers=orig_wn,
    resampled_wavenumbers=resampled_wn,
    nn_input_mat=real_nn_input_matrix,
    latent_bc_mat=real_latent_bc_matrix,
    conv_mat=real_processed_spectra.intensity_matrix,
    nn_mat=real_processed_matrix,
    num_samples=20
)

