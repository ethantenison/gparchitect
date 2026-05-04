"""
Time-varying kernel modules for GPArchitect — Tier 2 non-stationarity support.

Purpose:
    Implements GPyTorch kernel wrappers whose effective outputscale or lengthscale
    varies smoothly as a function of a designated time-like input dimension.

Role in pipeline:
    Natural language → GP DSL → Validation → **Model Builder** → Fit

Inputs:
    - base_kernel: a gpytorch.kernels.Kernel instance to wrap.
    - time_feature_index: the column of the input tensor that carries time.
    - target: TimeVaryingTarget.OUTPUTSCALE or TimeVaryingTarget.LENGTHSCALE.

Outputs:
    A gpytorch.kernels.Kernel instance with two additional learnable parameters
    (raw_tv_bias, raw_tv_slope) that control the time-dependent modulation.

Implementation choice:
    A linear-plus-softplus modulation was chosen for v1 because:
    - It has only two learnable scalar parameters, keeping optimization tractable.
    - softplus guarantees the modulation is strictly positive.
    - It is differentiable everywhere, compatible with torch.optim optimizers.
    - It is interpretable: bias controls the baseline scale; slope controls
      whether the hyperparameter grows or shrinks over time.
    - More expressive alternatives (neural-network modulation, Gibbs kernel) are
      left for future tiers; this conservative choice is safe and auditable.

    For OUTPUTSCALE target:
        k_tv(x_i, x_j) = s(t_i) * k_base(x_i, x_j) * s(t_j)
        where s(t) = softplus(bias + slope · t)

    For LENGTHSCALE target:
        The time column of each input is rescaled by 1 / l(t) before the base
        kernel is evaluated.  This changes the effective correlation along the
        time axis without modifying other dimensions.
        l(t) = softplus(bias + slope · t)

What this module does NOT do:
    - It does not handle multi-output or multi-task time variation.
    - It does not support non-linear parameterizations beyond the linear basis.
    - It does not modify the DSL or validation logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_SOFTPLUS_BETA = 1.0
_DEFAULT_TV_OUTPUTSCALE_BIAS_LIMIT = 4.0
_DEFAULT_TV_OUTPUTSCALE_SLOPE_LIMIT = 4.0


def build_time_varying_kernel(
    base_kernel: "object",
    time_feature_index: int,
    target: str,
    outputscale_bias_limit: float = _DEFAULT_TV_OUTPUTSCALE_BIAS_LIMIT,
    outputscale_slope_limit: float = _DEFAULT_TV_OUTPUTSCALE_SLOPE_LIMIT,
) -> "object":
    """Wrap a base GPyTorch kernel in a time-varying modulation module.

    Args:
        base_kernel: A gpytorch.kernels.Kernel instance built from the KernelSpec.
        time_feature_index: Zero-based column index of the time-like feature.
        target: Either "outputscale" or "lengthscale" (from TimeVaryingTarget values).
        outputscale_bias_limit: Max absolute bound for effective outputscale bias
            after tanh squashing (used only for outputscale target).
        outputscale_slope_limit: Max absolute bound for effective outputscale slope
            after tanh squashing (used only for outputscale target).

    Returns:
        A gpytorch.kernels.Kernel instance that adds a time-dependent modulation.

    Raises:
        ValueError: If target is not a recognised TimeVaryingTarget value.
    """
    if target == "outputscale":
        return _TimeVaryingOutputscaleKernel(
            base_kernel,
            time_feature_index,
            outputscale_bias_limit=outputscale_bias_limit,
            outputscale_slope_limit=outputscale_slope_limit,
        )
    if target == "lengthscale":
        return _TimeVaryingLengthscaleKernel(base_kernel, time_feature_index)
    raise ValueError(f"Unknown TimeVaryingTarget value: '{target}'. Expected 'outputscale' or 'lengthscale'.")


class _TimeVaryingOutputscaleKernel:
    """Non-stationary kernel with learned time-dependent amplitude.

    k_tv(x_i, x_j) = s(t_i) * k_base(x_i, x_j) * s(t_j)

    where s(t) = softplus(bias + slope · t) with learnable (bias, slope).

    This allows the signal variance to increase or decrease over time while keeping
    the base kernel's correlation structure intact.
    """

    def __new__(
        cls,
        base_kernel: "object",
        time_feature_index: int,
        *,
        outputscale_bias_limit: float,
        outputscale_slope_limit: float,
    ) -> "object":
        import gpytorch
        import torch

        class _Module(gpytorch.kernels.Kernel):
            has_lengthscale = False

            def __init__(self) -> None:
                super().__init__()
                self.base_kernel = base_kernel
                self.time_feature_index = time_feature_index
                self.outputscale_bias_limit = outputscale_bias_limit
                self.outputscale_slope_limit = outputscale_slope_limit
                # Initialize bias to 0 and slope to 0 → s(t) starts at softplus(0) ≈ 0.693
                self.register_parameter("raw_tv_bias", torch.nn.Parameter(torch.zeros(1)))
                self.register_parameter("raw_tv_slope", torch.nn.Parameter(torch.zeros(1)))

            def _bounded_params(self) -> tuple["torch.Tensor", "torch.Tensor"]:
                """Map unconstrained raw parameters to bounded effective values.

                Bounded parameters prevent runaway modulation growth in outputscale-only
                regimes where the optimizer can otherwise drive very large slopes.
                """
                bias = self.outputscale_bias_limit * torch.tanh(self.raw_tv_bias)
                slope = self.outputscale_slope_limit * torch.tanh(self.raw_tv_slope)
                return bias, slope

            def _modulation(self, x: "torch.Tensor") -> "torch.Tensor":
                """Compute s(t) = softplus(bias + slope · t) for each row in x."""
                t = x[..., self.time_feature_index]
                bias, slope = self._bounded_params()
                return torch.nn.functional.softplus(bias + slope * t)

            def forward(
                self,
                x1: "torch.Tensor",
                x2: "torch.Tensor",
                **kwargs: object,
            ) -> "torch.Tensor":
                s1 = self._modulation(x1)  # shape (..., N1)
                s2 = self._modulation(x2)  # shape (..., N2)
                k_base = self.base_kernel(x1, x2, **kwargs).to_dense()
                # Outer product of s1 and s2 gives the amplitude factor
                amplitude = s1.unsqueeze(-1) * s2.unsqueeze(-2)
                return amplitude * k_base

        return _Module()


class _TimeVaryingLengthscaleKernel:
    """Non-stationary kernel with learned time-dependent lengthscale along the time axis.

    The time column is rescaled by 1 / l(t) before base-kernel evaluation:

        x'_i[t] = x_i[t] / l(t_i)
        k_tv(x_i, x_j) = k_base(x'_i, x'_j)

    where l(t) = softplus(bias + slope · t) with learnable (bias, slope).

    This makes the correlation along the time axis shorter or longer as time
    progresses without changing other input dimensions.
    """

    def __new__(cls, base_kernel: "object", time_feature_index: int) -> "object":
        import gpytorch
        import torch

        class _Module(gpytorch.kernels.Kernel):
            has_lengthscale = False

            def __init__(self) -> None:
                super().__init__()
                self.base_kernel = base_kernel
                self.time_feature_index = time_feature_index
                self.register_parameter("raw_tv_bias", torch.nn.Parameter(torch.zeros(1)))
                self.register_parameter("raw_tv_slope", torch.nn.Parameter(torch.zeros(1)))

            def _effective_lengthscale(self, x: "torch.Tensor") -> "torch.Tensor":
                """Compute l(t) = softplus(bias + slope · t) for each row in x."""
                t = x[..., self.time_feature_index]
                return torch.nn.functional.softplus(self.raw_tv_bias + self.raw_tv_slope * t)

            def _warp_time(self, x: "torch.Tensor") -> "torch.Tensor":
                """Divide the time column by the local effective lengthscale."""
                l_t = self._effective_lengthscale(x)  # shape (..., N)
                x_warped = x.clone()
                x_warped[..., self.time_feature_index] = x[..., self.time_feature_index] / l_t
                return x_warped

            def forward(
                self,
                x1: "torch.Tensor",
                x2: "torch.Tensor",
                **kwargs: object,
            ) -> "torch.Tensor":
                x1_warped = self._warp_time(x1)
                x2_warped = self._warp_time(x2)
                return self.base_kernel(x1_warped, x2_warped, **kwargs).to_dense()

        return _Module()
