import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. THE LOG-COSH LOSS FUNCTION
# =====================================================================
class LogCoshLoss(nn.Module):
    """
    Logarithm of the hyperbolic cosine loss.
    A smooth, numerically stable approximation of MAE for robust optimization.
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Stable LogCosh formula: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device))
        )

# =====================================================================
# 2. CORE RESNET-INSPIRED 1D BLOCKS
# =====================================================================
class ConvBlock1D(nn.Module):
    """
    Block A: Used when stride > 1 or when channel sizes must change.
    Employs a 1D convolution with stride in the skip route to match shapes.
    """
    def __init__(self, in_channels, n_i1, n_i2, F_i, s_i):
        super(ConvBlock1D, self).__init__()
        # Main Route
        self.main_conv1 = nn.Conv1d(in_channels, n_i1, kernel_size=1, stride=s_i)
        self.bn1 = nn.BatchNorm1d(n_i1)
        self.main_conv2 = nn.Conv1d(n_i1, n_i1, kernel_size=F_i, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(n_i1)
        self.main_conv3 = nn.Conv1d(n_i1, n_i2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(n_i2)

        # Skip Route (aligned using 1x1 convolution with stride)
        self.skip_conv = nn.Conv1d(in_channels, n_i2, kernel_size=1, stride=s_i)
        self.skip_bn = nn.BatchNorm1d(n_i2)

    def forward(self, x):
        out = F.relu(self.bn1(self.main_conv1(x)))
        out = F.relu(self.bn2(self.main_conv2(out)))
        out = self.bn3(self.main_conv3(out))
        skip = self.skip_bn(self.skip_conv(x))
        return F.relu(out + skip)

class IdentityBlock1D(nn.Module):
    """
    Block B: Used when strides match (stride=1) and channel dimensions are identical.
    Uses a direct identity connection for the skip route.
    """
    def __init__(self, in_channels, n_i1, n_i2, F_i):
        super(IdentityBlock1D, self).__init__()
        # Main Route
        self.main_conv1 = nn.Conv1d(in_channels, n_i1, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(n_i1)
        self.main_conv2 = nn.Conv1d(n_i1, n_i1, kernel_size=F_i, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(n_i1)
        self.main_conv3 = nn.Conv1d(n_i1, n_i2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(n_i2)

    def forward(self, x):
        out = F.relu(self.bn1(self.main_conv1(x)))
        out = F.relu(self.bn2(self.main_conv2(out)))
        out = self.bn3(self.main_conv3(out))
        return F.relu(out + x)

# =====================================================================
# 3. SUB-NETWORKS (ENCODER AND DECODER)
# =====================================================================
class BaselineNet(nn.Module):
    """
    Sub-Network 1 (Encoder): Evaluates and extracts the baseline profile.
    Composed of 15 blocks grouped into 5 stages.
    """
    def __init__(self, input_length=1000):
        super(BaselineNet, self).__init__()
        self.input_length = input_length

        self.init_conv = nn.Conv1d(1, 8, kernel_size=3, padding='same', stride=1)
        self.init_bn = nn.BatchNorm1d(8)

        # 15 blocks grouped into 5 stages of 1 Conv Block & 2 Identity Blocks
        self.block1 = ConvBlock1D(in_channels=8, n_i1=32, n_i2=64, F_i=30, s_i=2)
        self.block2 = IdentityBlock1D(in_channels=64, n_i1=32, n_i2=64, F_i=30)
        self.block3 = IdentityBlock1D(in_channels=64, n_i1=32, n_i2=64, F_i=30)

        self.block4 = ConvBlock1D(in_channels=64, n_i1=32, n_i2=64, F_i=15, s_i=2)
        self.block5 = IdentityBlock1D(in_channels=64, n_i1=32, n_i2=64, F_i=15)
        self.block6 = IdentityBlock1D(in_channels=64, n_i1=32, n_i2=64, F_i=15)

        self.block7 = ConvBlock1D(in_channels=64, n_i1=64, n_i2=128, F_i=15, s_i=2)
        self.block8 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=15)
        self.block9 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=15)

        self.block10 = ConvBlock1D(in_channels=128, n_i1=128, n_i2=256, F_i=9, s_i=2)
        self.block11 = IdentityBlock1D(in_channels=256, n_i1=128, n_i2=256, F_i=9)
        self.block12 = IdentityBlock1D(in_channels=256, n_i1=128, n_i2=256, F_i=9)

        self.block13 = ConvBlock1D(in_channels=256, n_i1=256, n_i2=512, F_i=3, s_i=2)
        self.block14 = IdentityBlock1D(in_channels=512, n_i1=256, n_i2=512, F_i=3)
        self.block15 = IdentityBlock1D(in_channels=512, n_i1=256, n_i2=512, F_i=3)

        # Calculate dynamic flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy_out = self._forward_features(dummy)
            self.flat_features = dummy_out.numel()

        self.fc = nn.Linear(self.flat_features, input_length)

    def _forward_features(self, x):
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        return x

    def forward(self, x):
        features = self._forward_features(x)
        features = torch.flatten(features, 1)
        baseline_corrected = self.fc(features)
        return baseline_corrected.unsqueeze(1) # Return as (N, 1, input_length)


class DenoisingNet(nn.Module):
    """
    Sub-Network 2 (Decoder): Filters noise and spike artifacts.
    Starts with a standard 1D Convolution followed by 3 residual stages.
    """
    def __init__(self, input_length=1000):
        super(DenoisingNet, self).__init__()
        self.input_length = input_length

        self.init_conv = nn.Conv1d(1, 8, kernel_size=3, padding='same', stride=1)
        self.init_bn = nn.BatchNorm1d(8)

        # 3 Stages of 1 Conv Block and 2 Identity Blocks
        self.stage1_block1 = ConvBlock1D(in_channels=8, n_i1=64, n_i2=128, F_i=3, s_i=2)
        self.stage1_block2 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)
        self.stage1_block3 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)

        self.stage2_block1 = ConvBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3, s_i=2)
        self.stage2_block2 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)
        self.stage2_block3 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)

        self.stage3_block1 = ConvBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3, s_i=2)
        self.stage3_block2 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)
        self.stage3_block3 = IdentityBlock1D(in_channels=128, n_i1=64, n_i2=128, F_i=3)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy_out = self._forward_features(dummy)
            self.flat_features = dummy_out.numel()

        self.fc = nn.Linear(self.flat_features, input_length)

    def _forward_features(self, x):
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        return x

    def forward(self, x):
        features = self._forward_features(x)
        features = torch.flatten(features, 1)
        denoised = self.fc(features)
        return denoised.unsqueeze(1)

# =====================================================================
# 4. UNIFIED DUAL-SUPERVISED ARCHITECTURE (SCENARIO C)
# =====================================================================
class DualSupervisedNet(nn.Module):
    """
    Cooperative Double-ResNet Architecture.
    Trains the encoder to remove baselines, and the decoder to denoise.
    """
    def __init__(self, input_length=1000):
        super(DualSupervisedNet, self).__init__()
        self.encoder = BaselineNet(input_length=input_length)  # Sub-Net 1
        self.decoder = DenoisingNet(input_length=input_length) # Sub-Net 2

    def forward(self, x):
        # x: Raw, baseline-contaminated, noisy spectra (N, 1, input_length)

        # Step 1: Encoder directly predicts the baseline corrected spectra
        latent_baseline_corrected = self.encoder(x)

        # Step 2: Decoder processes the intermediate state to output clean spectra
        clean_reconstructed = self.decoder(latent_baseline_corrected)

        return latent_baseline_corrected, clean_reconstructed

