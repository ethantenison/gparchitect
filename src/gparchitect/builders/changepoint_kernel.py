"""
Changepoint kernel for GPArchitect — time-driven non-stationary GP support.

Purpose:
    Provides a GPyTorch kernel that blends two sub-kernels via a learnable sigmoid
    weighting function, enabling the GP to represent regime changes over a time-like
    input dimension.

Role in pipeline:
    Natural language → GP DSL → Validation → **Model Builder** (this module) → Fit

Inputs:
    - kernel_before: A GPyTorch kernel active before the changepoint.
    - kernel_after: A GPyTorch kernel active after the changepoint.
    - changepoint_location: Initial value for the sigmoid inflection point.
    - changepoint_steepness: Initial value for the sigmoid slope (controls sharpness
      of the transition).

Outputs:
    A gpytorch.kernels.Kernel instance whose covariance is:

        K_cp(x, x') = σ(x) · K_before(x, x') · σ(x')
                    + (1 − σ(x)) · K_after(x, x') · (1 − σ(x'))

    where σ(t) = sigmoid((t − location) / steepness).

Non-obvious design decisions:
    - Both location and steepness are unconstrained learnable parameters (optimised
      during MLL maximisation) unless the spec sets them to fixed values.
    - The kernel is designed to act on a single time-like scalar input (the active_dim
      passed by the builder).  ARD is not meaningful here and is not applied.
    - sub-kernels are kept as child modules so that their parameters appear in
      model.named_parameters() and are optimised jointly.

What this module does NOT do:
    - It does not select which input dimension is treated as time — that is the
      builder's responsibility via the active_dims mechanism.
    - It does not wrap itself in a ScaleKernel; the builder handles scaling.
"""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel


class ChangepointKernel(Kernel):
    """A kernel that smoothly transitions between two sub-kernels at a learned changepoint.

    The covariance is:

        K_cp(x, x') = σ(x) · K1(x, x') · σ(x') + (1 − σ(x)) · K2(x, x') · (1 − σ(x'))

    where σ(t) = sigmoid((t − location) / steepness) is a learnable sigmoid gate and
    K1, K2 are the "before" and "after" sub-kernels respectively.

    Attributes:
        kernel_before: Sub-kernel dominating before the changepoint.
        kernel_after: Sub-kernel dominating after the changepoint.
        raw_location: Unconstrained learnable parameter for the changepoint location.
        raw_log_steepness: Unconstrained learnable parameter (log of steepness so
            steepness remains positive).
    """

    has_lengthscale = False

    def __init__(
        self,
        kernel_before: Kernel,
        kernel_after: Kernel,
        changepoint_location: float = 0.5,
        changepoint_steepness: float = 1.0,
        **kwargs: object,
    ) -> None:
        """Initialise the changepoint kernel.

        Args:
            kernel_before: The GPyTorch kernel to use before the changepoint.
            kernel_after: The GPyTorch kernel to use after the changepoint.
            changepoint_location: Initial location of the sigmoid inflection point.
            changepoint_steepness: Initial sigmoid steepness.  Must be > 0.
            **kwargs: Forwarded to the parent Kernel constructor (e.g., active_dims).
        """
        super().__init__(**kwargs)

        if changepoint_steepness <= 0:
            raise ValueError(f"changepoint_steepness must be > 0, got {changepoint_steepness}.")

        self.kernel_before = kernel_before
        self.kernel_after = kernel_after

        self.register_parameter(
            "raw_location",
            torch.nn.Parameter(torch.tensor(changepoint_location, dtype=torch.float64)),
        )
        self.register_parameter(
            "raw_log_steepness",
            torch.nn.Parameter(torch.tensor(changepoint_steepness, dtype=torch.float64).log()),
        )

    @property
    def location(self) -> torch.Tensor:
        """Changepoint location (unconstrained real)."""
        return self.raw_location

    @property
    def steepness(self) -> torch.Tensor:
        """Changepoint steepness (positive, via exp transform)."""
        return self.raw_log_steepness.exp()

    def _sigmoid_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the sigmoid gate values for a batch of inputs.

        Args:
            x: Tensor of shape (..., 1) — the time-like feature values.

        Returns:
            Tensor of shape (...,) with values in (0, 1).
        """
        return torch.sigmoid((x.squeeze(-1) - self.location) / self.steepness)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **params: object,
    ) -> torch.Tensor:
        """Compute the changepoint covariance matrix.

        Args:
            x1: Input tensor of shape (..., N, D).
            x2: Input tensor of shape (..., M, D).
            **params: Additional keyword arguments forwarded to sub-kernels.

        Returns:
            Covariance tensor of shape (..., N, M).
        """
        gate1 = self._sigmoid_gate(x1)  # (..., N)
        gate2 = self._sigmoid_gate(x2)  # (..., M)

        # Sub-kernels return LinearOperator objects in modern GPyTorch.
        # Materialize to dense tensors for gate-weighted arithmetic below.
        k_before = self.kernel_before(x1, x2, **params).to_dense()
        k_after = self.kernel_after(x1, x2, **params).to_dense()

        # outer products of gate values
        gate_before = gate1.unsqueeze(-1) * gate2.unsqueeze(-2)  # (..., N, M)
        gate_after = (1 - gate1).unsqueeze(-1) * (1 - gate2).unsqueeze(-2)  # (..., N, M)

        return gate_before * k_before + gate_after * k_after
