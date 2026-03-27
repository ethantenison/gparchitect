"""
Fitting and prediction validation for GPArchitect.

Purpose:
    Fits a BoTorch GP model using standard BoTorch procedures and runs minimal
    prediction checks to detect fitting failures.

Role in pipeline:
    Natural language → GP DSL → Validation → Model Builder → **Fit** → Validation → Recovery

Inputs:
    - model: A BoTorch GP model instance.
    - train_X: Input training tensor.
    - train_Y: Output training tensor.
    - max_iter: Maximum number of optimisation iterations.

Outputs:
    FitResult — a dataclass holding success status, the optimised model,
    the final marginal log-likelihood, and any error message.

Non-obvious design decisions:
    - Uses botorch.fit.fit_gpytorch_mll (the standard BoTorch fitting utility).
        - Chooses the marginal log-likelihood class from the model type so independent
            ModelListGP outputs fit through SumMarginalLogLikelihood.
    - A brief forward pass with train_X is used as the prediction validation check.
    - Fitting errors are caught and returned in FitResult rather than re-raised,
      enabling the revision/recovery loop to handle them gracefully.

What this module does NOT do:
    - It does not revise the DSL on failure (see revision module).
    - It does not log revision history (see logging module).
    - It does not perform cross-validation or out-of-sample evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Result of a model fitting attempt.

    Attributes:
        success: Whether fitting and prediction check succeeded.
        model: The fitted model (or the unfitted model on failure).
        mll_value: Final marginal log-likelihood value, or None on failure.
        error_message: Error message if fitting failed, otherwise empty string.
        train_X: The training inputs used for this fit.
        train_Y: The training outputs used for this fit.
    """

    success: bool
    model: object
    mll_value: float | None
    error_message: str = ""
    train_X: "torch.Tensor | None" = field(default=None, repr=False)
    train_Y: "torch.Tensor | None" = field(default=None, repr=False)


def fit_and_validate(
    model,  # noqa: ANN001
    train_X: "torch.Tensor",
    train_Y: "torch.Tensor",
    max_iter: int = 100,
) -> FitResult:
    """Fit a BoTorch GP model and run a basic prediction check.

    Args:
        model: A BoTorch GP model instance (not yet fitted).
        train_X: Input training tensor of shape (N, D).
        train_Y: Output training tensor of shape (N, M).
        max_iter: Maximum number of L-BFGS optimisation iterations.

    Returns:
        FitResult indicating success or failure, with the (possibly fitted) model.
    """
    try:
        import gpytorch
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.fit import fit_gpytorch_mll

        if isinstance(model, ModelListGP):
            mll = gpytorch.mlls.SumMarginalLogLikelihood(model.likelihood, model)
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        mll_value = _compute_mll(mll, train_X, train_Y)

        # Basic prediction check — forward pass with training data
        model.eval()
        model.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            posterior = model.posterior(train_X)
            _ = posterior.mean  # ensure no runtime error

        logger.info("Fitting succeeded: mll=%.4f", mll_value if mll_value is not None else float("nan"))
        return FitResult(success=True, model=model, mll_value=mll_value, train_X=train_X, train_Y=train_Y)

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.warning("Fitting failed: %s", error_message)
        return FitResult(
            success=False,
            model=model,
            mll_value=None,
            error_message=error_message,
            train_X=train_X,
            train_Y=train_Y,
        )


def _compute_mll(mll, train_X: "torch.Tensor", train_Y: "torch.Tensor") -> float | None:  # noqa: ANN001
    """Compute the marginal log-likelihood on the training data.

    Args:
        mll: The ExactMarginalLogLikelihood object.
        train_X: Training inputs.
        train_Y: Training outputs.

    Returns:
        The MLL value as a float, or None if computation fails.
    """
    try:
        from botorch.models.model_list_gp_regression import ModelListGP
        import torch

        mll.train()
        if isinstance(mll.model, ModelListGP):
            output = mll.model(*[train_X for _ in mll.model.models])
            targets = [submodel.train_targets for submodel in mll.model.models]
            loss = -mll(output, targets)
        else:
            output = mll.model(train_X)
            loss = -mll(output, train_Y.squeeze(-1))
        return float(loss.item())
    except Exception as exc:
        logger.debug("MLL computation failed: %s", exc)
        return None
