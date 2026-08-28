import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. POLYNOMIAL BASELINE ESTIMATOR (100% Dynamic Input Length Agnostic)
# ============================================================================
class PolynomialBaselineEstimator(nn.Module):
    """
    Predicts (poly_order + 1) polynomial coefficients from raw spectra of ANY length.
    Dynamically constructs the baseline across [-1.0, 1.0] for whatever length L is passed.
    """
    def __init__(self, poly_order: int = 15):
        super().__init__()
        self.poly_order = poly_order

        # Feature extractor uses AdaptiveAvgPool1d(1) to collapse arbitrary length L -> 1
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Output: (Batch, 128, 1) for ANY input length L
        )

        # FC head predicts polynomial coefficients (c0, c1, ..., c_poly_order)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, poly_order + 1)
        )

    def forward(self, x: torch.Tensor):
        # x shape: (Batch, 1, L)
        L = x.shape[-1]
        
        # 1. Extract features & predict coefficients
        feat = self.encoder(x).squeeze(-1)  # (Batch, 128)
        coeffs = self.fc(feat)              # (Batch, poly_order + 1)

        # 2. Dynamically create coordinate grid over [-1.0, 1.0] for length L
        grid = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, 1, L)

        # 3. Evaluate polynomial via Horner's Method:
        # B(x) = c0 + x * (c1 + x * (c2 + ... + x * c_poly_order))
        # Numerically stable and memory-efficient with zero fixed buffers.
        baseline = coeffs[:, -1:].unsqueeze(-1)  # Start with highest degree coeff
        for i in range(self.poly_order - 1, -1, -1):
            c_i = coeffs[:, i : i + 1].unsqueeze(-1)  # (Batch, 1, 1)
            baseline = baseline * grid + c_i

        return baseline, coeffs


# ============================================================================
# 2. FILTER PARAMETER PREDICTOR (Fully Convolutional, Input-Size Agnostic)
# ============================================================================
class DilatedResidualBlock1D(nn.Module):
    """Preserves full sequence length L for any L."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class FilterParameterPredictor(nn.Module):
    """
    Fully convolutional 1D dilated residual network.
    Predicts [sigma] per bin for any sequence length L.
    """
    def __init__(self, channels: int = 32, min_sigma: float = 0.2, max_sigma: float = 6.0):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        self.in_conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        # Receptive field expansion with dilation (keeps sequence length L)
        self.res_blocks = nn.Sequential(
            DilatedResidualBlock1D(channels, dilation=1),
            DilatedResidualBlock1D(channels, dilation=2),
            DilatedResidualBlock1D(channels, dilation=4),
            DilatedResidualBlock1D(channels, dilation=8),
        )

        # 1x1 conv -> 1 physical parameter per bin (sigma)
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (Batch, 1, L)
        feat = self.in_conv(x)
        feat = self.res_blocks(feat)
        params = self.out_conv(feat)  # (Batch, 1, L)

        # Channel 0 -> Sigma (width)
        sigma = self.min_sigma + torch.sigmoid(params) * (self.max_sigma - self.min_sigma)

        return sigma


# ============================================================================
# 3. ADAPTIVE GAUSSIAN FILTER LAYER (Input-Size Agnostic)
# ============================================================================
class AdaptiveGaussianFilter1D(nn.Module):
    """
    Standard Gaussian filter with fixed exponent=2 and unit amplitude (A=1),
    adaptive solely in per-bin width (sigma).
    """
    def __init__(self, kernel_size: int = 31):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Offsets: (1, K, 1)
        diff = self.coords.view(1, -1, 1)

        # Standard Gaussian weights with fixed exponent 2: (Batch, K, L)
        weights = torch.exp(-0.5 * ((diff / (sigma + 1e-8)) ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Extract sliding windows for any length L: (Batch, K, L)
        patches = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), padding=(self.padding, 0)).squeeze(-1)

        # Convolve: (Batch, 1, L)
        smoothed = (patches * weights).sum(dim=1, keepdim=True)

        return smoothed


# Alias for backward compatibility if needed
AdaptiveGenGaussianFilter1D = AdaptiveGaussianFilter1D


# ============================================================================
# 4. FULL INPUT-SIZE AGNOSTIC MODEL: PolyGaussNet
# ============================================================================
class PolyGaussNet(nn.Module):
    """
    100% Input-Size Agnostic Raman Preprocessing Network (V2: Sigma-only Adaptive).
    Accepts (Batch, 1, L) or (Batch, L) for ANY length L.
    """
    def __init__(self, poly_order: int = 15, filter_kernel_size: int = 31):
        super().__init__()
        self.baseline_net = PolynomialBaselineEstimator(poly_order=poly_order)
        self.filter_param_net = FilterParameterPredictor()
        self.filter_layer = AdaptiveGaussianFilter1D(kernel_size=filter_kernel_size)

    def forward(self, raw_spectrum: torch.Tensor):
        # Allow 2D input (Batch, L) or 3D input (Batch, 1, L)
        is_2d = (raw_spectrum.ndim == 2)
        if is_2d:
            raw_spectrum = raw_spectrum.unsqueeze(1)

        # Part 1: Baseline Estimation & Subtraction
        pred_baseline, coeffs = self.baseline_net(raw_spectrum)
        baseline_corrected = raw_spectrum - pred_baseline

        # Part 2: Predict Per-Bin Physical Parameter (sigma only)
        sigma = self.filter_param_net(baseline_corrected)

        # Part 3: Adaptive Filtering (exponent=2, amplitude=1)
        clean_spectrum = self.filter_layer(baseline_corrected, sigma)

        latent_params = {
            'coeffs': coeffs,
            'sigma': sigma if not is_2d else sigma.squeeze(1)
        }

        if is_2d:
            clean_spectrum = clean_spectrum.squeeze(1)
            pred_baseline = pred_baseline.squeeze(1)
            baseline_corrected = baseline_corrected.squeeze(1)

        return clean_spectrum, pred_baseline, baseline_corrected, latent_params
