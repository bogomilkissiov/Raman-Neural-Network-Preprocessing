"""
PolyGaussNet V3-Len (Ablation Variant)
--------------------------------------
Identical to PolyGaussNet V3 in every way EXCEPT that the baseline FC head receives
one extra input: the normalized sequence length scalar (L / 1600.0), exactly as in V3.1.

Purpose: isolate how much of the V3 -> V3.1 improvement comes purely from length
conditioning vs. from the rest of the V3.1 architecture (new baseline encoder +
re-balanced filter predictor).

Initialization detail (controlled ablation):
  - The V3 model is constructed first (consuming the same RNG as plain V3), so under
    the same seed the encoder, FC layers and filter predictor get IDENTICAL weights to V3.
  - The first FC layer is widened from 128 -> 129 inputs by copying V3's weights and
    adding a ZERO-initialized column for the length scalar.
  => At initialization V3-Len computes exactly the same function as V3. Any difference
     after training is attributable to the length input alone.
"""

import torch
import torch.nn as nn

from polygaussnet3 import (
    PolyGaussNet as _PolyGaussNet3,
    PolynomialBaselineEstimator as _PolynomialBaselineEstimator3,
)

LENGTH_NORM = 1600.0  # Same normalization constant as V3.1


class LengthAwarePolynomialBaselineEstimator(nn.Module):
    """V3 baseline estimator + normalized-length scalar injected into the FC head."""

    def __init__(self, base: _PolynomialBaselineEstimator3):
        super().__init__()
        self.poly_order = base.poly_order
        self.encoder = base.encoder  # Reuse V3 encoder (identical weights under same seed)

        old_first = base.fc[0]  # nn.Linear(128, 128)
        new_first = nn.Linear(old_first.in_features + 1, old_first.out_features)
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, : old_first.in_features].copy_(old_first.weight)
            new_first.bias.copy_(old_first.bias)

        # Keep the remaining V3 FC layers untouched
        self.fc = nn.Sequential(new_first, *list(base.fc.children())[1:])

    def forward(self, x: torch.Tensor):
        L = x.shape[-1]
        feat = self.encoder(x).squeeze(-1)  # (Batch, 128)

        len_scalar = torch.full((x.shape[0], 1), L / LENGTH_NORM, device=x.device, dtype=x.dtype)
        coeffs = self.fc(torch.cat([feat, len_scalar], dim=1))  # (Batch, poly_order + 1)

        grid = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, 1, L)
        baseline = coeffs[:, -1:].unsqueeze(-1)
        for i in range(self.poly_order - 1, -1, -1):
            c_i = coeffs[:, i : i + 1].unsqueeze(-1)
            baseline = baseline * grid + c_i

        return baseline, coeffs


class PolyGaussNet(_PolyGaussNet3):
    """PolyGaussNet V3 with a length-aware baseline head (only change vs. V3)."""

    def __init__(
        self,
        poly_order: int = 15,
        filter_kernel_size: int = 31,
        min_amplitude: float = 0.85,
        max_amplitude: float = 1.15,
    ):
        super().__init__(
            poly_order=poly_order,
            filter_kernel_size=filter_kernel_size,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
        )
        self.baseline_net = LengthAwarePolynomialBaselineEstimator(self.baseline_net)
