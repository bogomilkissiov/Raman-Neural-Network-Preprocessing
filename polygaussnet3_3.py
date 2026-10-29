import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. DILATED RESIDUAL BLOCK (Used for Wide Receptive Field Feature Extraction)
# ============================================================================
class DilatedResidualBlock1D(nn.Module):
    """
    Dilated residual block with reflection padding.
    Maintains sequence length while expanding physical receptive field exponentially.
    """
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


# ============================================================================
# 2. POLYNOMIAL BASELINE ESTIMATOR (V3.3: 4-Quadrant Spatial Pooling + Funnel)
# ============================================================================
class PolynomialBaselineEstimator(nn.Module):
    """
    Predicts (poly_order + 1) polynomial coefficients from raw spectra of ANY length.
    
    Architecture (PolyGaussNet V3.3: 4-Quadrant Spatial Pooling & Funnel):
    - Retains lightweight 32-channel convolutional trunk from V3.1 (0 extra conv weights).
    - Strided stem (4x downsampling): 1 -> 32 -> 32 with stride 2 each.
    - 3 Dilated Residual Blocks (dilations 1, 2, 4) with 32 channels (251-bin receptive field).
    - 4-Quadrant Spatial Pooling: AdaptiveAvgPool1d(4) preserves spatial geometry
      [Left 0-25%, Center-Left 25-50%, Center-Right 50-75%, Right 75-100%].
    - Flatten across channels & spatial zones: (Batch, 32, 4) -> (Batch, 128).
    - Length injection: concatenate normalized sequence length scalar (L / 1600.0) -> (Batch, 129).
    - FC regression head: clean contracting funnel 129 -> 64 -> (poly_order + 1).
      Directly provides spatial gradient (Right - Left) and curvature (Center - Ends)
      to the FC head with no artificial channel expansion.
    - Dynamically constructs the baseline across [-1.0, 1.0] via Horner's Method for any length L.
    """
    def __init__(self, poly_order: int = 7, channels: int = 32, num_quadrants: int = 4):
        super().__init__()
        self.poly_order = poly_order
        self.channels = channels
        self.num_quadrants = num_quadrants

        # 4x Strided Stem (Low-pass filter for baseline)
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        # 3 Dilated residual blocks on downsampled feature map
        # Stride 4 * dilations [1, 2, 4] -> Receptive Field = 251 bins
        self.res_blocks = nn.Sequential(
            DilatedResidualBlock1D(channels, dilation=1),
            DilatedResidualBlock1D(channels, dilation=2),
            DilatedResidualBlock1D(channels, dilation=4),
        )

        # 4-quadrant spatial pooling: divides spectrum into 4 spatial zones
        self.pool = nn.AdaptiveAvgPool1d(num_quadrants)  # (Batch, channels, num_quadrants)

        # Clean contracting funnel FC head: (channels * num_quadrants + 1) -> 64 -> (poly_order + 1)
        # For channels=32, quadrants=4: 128 + 1 = 129 -> 64 -> 8
        in_features = channels * num_quadrants + 1
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, poly_order + 1)
        )

    def _adaptive_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """
        4-quadrant pooling with seamless MPS support.
        PyTorch MPS has an open limitation (issue #96056) where F.adaptive_avg_pool1d
        fails if input length is not divisible by output_size. When running on MPS with
        non-divisible sequence lengths, this evaluates the exact mathematical pooling
        slices natively on device.
        """
        L_feat = feat.shape[-1]
        if feat.device.type != "mps" or (L_feat % self.num_quadrants == 0):
            return self.pool(feat)
        chunks = [
            feat[..., (k * L_feat) // self.num_quadrants : ((k + 1) * L_feat + self.num_quadrants - 1) // self.num_quadrants].mean(dim=-1, keepdim=True)
            for k in range(self.num_quadrants)
        ]
        return torch.cat(chunks, dim=-1)

    def forward(self, x: torch.Tensor):
        # x shape: (Batch, 1, L)
        L = x.shape[-1]
        
        # 1. Multi-scale feature extraction with wide receptive field (251 bins)
        feat = self.stem(x)
        feat = self.res_blocks(feat)

        # 2. 4-quadrant spatial pooling & flattening across channels + quadrants
        # (Batch, channels, 4) -> (Batch, channels * 4) e.g. (Batch, 128)
        pooled = torch.flatten(self._adaptive_pool(feat), start_dim=1)

        # 3. Inject normalized sequence length scalar into FC head
        len_scalar = torch.full((x.shape[0], 1), L / 1600.0, device=x.device, dtype=x.dtype)
        feat_with_len = torch.cat([pooled, len_scalar], dim=1)  # (Batch, 129)
        coeffs = self.fc(feat_with_len)                         # (Batch, poly_order + 1)

        # 4. Dynamically create coordinate grid over [-1.0, 1.0] for length L
        grid = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, 1, L)

        # 5. Evaluate polynomial via Horner's Method:
        # B(x) = c0 + x * (c1 + x * (c2 + ... + x * c_poly_order))
        baseline = coeffs[:, -1:].unsqueeze(-1)
        for i in range(self.poly_order - 1, -1, -1):
            c_i = coeffs[:, i : i + 1].unsqueeze(-1)
            baseline = baseline * grid + c_i

        return baseline, coeffs


# ============================================================================
# 3. FILTER PARAMETER PREDICTOR (Local Multi-Scale CNN, Stride=1, Agnostic)
# ============================================================================
class FilterParameterPredictor(nn.Module):
    """
    Local multi-scale convolutional network (stride=1, reflection padding).
    Receptive field: 15 + 10 + 6 = 31 bins (matches 1-20 bin peak widths and 31-bin Gaussian kernel).
    Predicts 2 physical parameters per bin: [sigma, amplitude] for any sequence length L.
    Exact 3-layer architecture from V3.1 retained as-is.
    """
    def __init__(
        self,
        channels: int = 32,
        min_sigma: float = 0.2,
        max_sigma: float = 6.0,
        min_amplitude: float = 0.85,
        max_amplitude: float = 1.15
    ):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

        # 3-layer convolutional feature extractor with stride=1 (preserves exact sequence length L)
        self.conv1 = nn.Conv1d(1, channels, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(channels * 2)
        self.conv3 = nn.Conv1d(channels * 2, channels * 2, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(channels * 2)
        self.act = nn.GELU()

        # 1x1 conv -> 2 physical parameters per bin (sigma, amplitude)
        self.out_conv = nn.Conv1d(channels * 2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (Batch, 1, L)
        feat = self.act(self.bn1(self.conv1(x)))
        feat = self.act(self.bn2(self.conv2(feat)))
        feat = self.act(self.bn3(self.conv3(feat)))
        params = self.out_conv(feat)  # (Batch, 2, L)

        # Channel 0 -> Sigma (filter width)
        sigma = self.min_sigma + torch.sigmoid(params[:, 0:1, :]) * (self.max_sigma - self.min_sigma)

        # Channel 1 -> Bounded Amplitude Modulation
        amplitude = self.min_amplitude + torch.sigmoid(params[:, 1:2, :]) * (self.max_amplitude - self.min_amplitude)

        return sigma, amplitude


# ============================================================================
# 4. ADAPTIVE GAUSSIAN FILTER LAYER (Input-Size Agnostic)
# ============================================================================
class AdaptiveGaussianFilter1D(nn.Module):
    """
    Gaussian filter with fixed exponent=2 and per-bin adaptive width (sigma)
    along with bounded per-bin amplitude modulation.
    """
    def __init__(self, kernel_size: int = 31):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        # Offsets: (1, K, 1)
        diff = self.coords.view(1, -1, 1)

        # Standard Gaussian weights with fixed exponent 2: (Batch, K, L)
        weights = torch.exp(-0.5 * ((diff / (sigma + 1e-8)) ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Extract sliding windows for any length L: (Batch, K, L) with zero padding
        patches = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), padding=(self.padding, 0)).squeeze(-1)

        # Convolve: (Batch, 1, L)
        smoothed = (patches * weights).sum(dim=1, keepdim=True)

        # Apply bounded amplitude modulation
        return smoothed * amplitude


# Alias for backward compatibility if needed
AdaptiveGenGaussianFilter1D = AdaptiveGaussianFilter1D


# ============================================================================
# 5. FULL MODEL: PolyGaussNet (V3.3: 4-Quadrant Pooling + Clean Funnel)
# ============================================================================
class PolyGaussNet(nn.Module):
    """
    Input-Size Agnostic Raman Preprocessing Network (V3.3).
    - Baseline Estimator: 32-channel trunk + 4-quadrant spatial pooling + clean contracting funnel (129 -> 64 -> 8).
    - Filter Predictor: 31-bin receptive field matching local Raman peak scales (1-20 bins).
    - Zero padding (standard matching V3/V3.1).
    Accepts (Batch, 1, L) or (Batch, L) for ANY length L.
    """
    def __init__(
        self,
        poly_order: int = 7,
        filter_kernel_size: int = 31,
        min_amplitude: float = 0.85,
        max_amplitude: float = 1.15,
        baseline_channels: int = 32,
        filter_channels: int = 32
    ):
        super().__init__()
        self.baseline_net = PolynomialBaselineEstimator(poly_order=poly_order, channels=baseline_channels)
        self.filter_param_net = FilterParameterPredictor(
            channels=filter_channels,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude
        )
        self.filter_layer = AdaptiveGaussianFilter1D(kernel_size=filter_kernel_size)

    def forward(self, raw_spectrum: torch.Tensor):
        # Allow 2D input (Batch, L) or 3D input (Batch, 1, L)
        is_2d = (raw_spectrum.ndim == 2)
        if is_2d:
            raw_spectrum = raw_spectrum.unsqueeze(1)

        # Part 1: Baseline Estimation & Subtraction
        pred_baseline, coeffs = self.baseline_net(raw_spectrum)
        baseline_corrected = raw_spectrum - pred_baseline

        # Part 2: Predict Per-Bin Physical Parameters (sigma + bounded amplitude)
        sigma, amplitude = self.filter_param_net(baseline_corrected)

        # Part 3: Adaptive Filtering with Bounded Amplitude Modulation
        clean_spectrum = self.filter_layer(baseline_corrected, sigma, amplitude)

        latent_params = {
            'coeffs': coeffs,
            'sigma': sigma if not is_2d else sigma.squeeze(1),
            'amplitude': amplitude if not is_2d else amplitude.squeeze(1)
        }

        if is_2d:
            clean_spectrum = clean_spectrum.squeeze(1)
            pred_baseline = pred_baseline.squeeze(1)
            baseline_corrected = baseline_corrected.squeeze(1)

        return clean_spectrum, pred_baseline, baseline_corrected, latent_params
