"""
PolyGaussNet V3.3-NoLen (Ablation Variant)
------------------------------------------
Identical to PolyGaussNet V3.3 in every way EXCEPT that the baseline FC head does NOT
receive the normalized sequence length scalar (L / 1600.0).

    V3.3       : [32 ch x 4 quadrants] + [L/1600] -> 129 -> 64 -> (poly_order + 1)
    V3.3-NoLen : [32 ch x 4 quadrants]            -> 128 -> 64 -> (poly_order + 1)

Purpose: isolate how much of V3.3's performance depends on length conditioning vs. the
4-quadrant spatial pooling. Mirror image of the V3-Len ablation (which ADDS the scalar to V3.0).

Initialization detail (controlled ablation):
  - The full V3.3 model is constructed first (consuming the same RNG as plain V3.3), so under
    the same seed the trunk, FC layers and filter predictor get IDENTICAL weights to V3.3.
  - The first FC layer is narrowed from 129 -> 128 inputs by copying V3.3's weights and
    dropping the column that multiplied the length scalar.
  => The only difference vs V3.3 at initialization is the removed length-scalar column.
"""

import torch
import torch.nn as nn

from polygaussnet3_3 import (
    PolyGaussNet as _PolyGaussNet3_3,
    PolynomialBaselineEstimator as _PolynomialBaselineEstimator3_3,
)


class NoLengthPolynomialBaselineEstimator(nn.Module):
    """V3.3 baseline estimator (4-quadrant pooling + funnel) WITHOUT the length scalar."""

    def __init__(self, base: _PolynomialBaselineEstimator3_3):
        super().__init__()
        self.poly_order = base.poly_order
        self.channels = base.channels
        self.num_quadrants = base.num_quadrants

        # Reuse V3.3 trunk + pooling (identical weights under same seed)
        self.stem = base.stem
        self.res_blocks = base.res_blocks
        self.pool = base.pool

        old_first = base.fc[0]  # nn.Linear(channels * num_quadrants + 1, 64)
        in_features = self.channels * self.num_quadrants  # 128 (no length scalar)
        assert old_first.in_features == in_features + 1, "Unexpected V3.3 FC input width"
        new_first = nn.Linear(in_features, old_first.out_features)
        with torch.no_grad():
            # Length scalar was concatenated LAST -> drop the last weight column
            new_first.weight.copy_(old_first.weight[:, :in_features])
            new_first.bias.copy_(old_first.bias)

        # Keep the remaining V3.3 FC layers untouched
        self.fc = nn.Sequential(new_first, *list(base.fc.children())[1:])

    # Same MPS-safe 4-quadrant pooling as V3.3
    _adaptive_pool = _PolynomialBaselineEstimator3_3._adaptive_pool

    def forward(self, x: torch.Tensor):
        # x shape: (Batch, 1, L)
        L = x.shape[-1]

        # 1. Multi-scale feature extraction (251-bin receptive field)
        feat = self.stem(x)
        feat = self.res_blocks(feat)

        # 2. 4-quadrant spatial pooling & flattening -> (Batch, 128). NO length injection.
        pooled = torch.flatten(self._adaptive_pool(feat), start_dim=1)
        coeffs = self.fc(pooled)  # (Batch, poly_order + 1)

        # 3. Dynamically create coordinate grid over [-1.0, 1.0] for length L
        grid = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, 1, L)

        # 4. Evaluate polynomial via Horner's Method
        baseline = coeffs[:, -1:].unsqueeze(-1)
        for i in range(self.poly_order - 1, -1, -1):
            c_i = coeffs[:, i : i + 1].unsqueeze(-1)
            baseline = baseline * grid + c_i

        return baseline, coeffs


class PolyGaussNet(_PolyGaussNet3_3):
    """PolyGaussNet V3.3 with the length scalar removed from the baseline head (only change vs. V3.3)."""

    def __init__(
        self,
        poly_order: int = 7,
        filter_kernel_size: int = 31,
        min_amplitude: float = 0.85,
        max_amplitude: float = 1.15,
        baseline_channels: int = 32,
        filter_channels: int = 32,
    ):
        super().__init__(
            poly_order=poly_order,
            filter_kernel_size=filter_kernel_size,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            baseline_channels=baseline_channels,
            filter_channels=filter_channels,
        )
        self.baseline_net = NoLengthPolynomialBaselineEstimator(self.baseline_net)
