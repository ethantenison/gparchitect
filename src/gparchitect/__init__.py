"""
GPArchitect — build BoTorch GP models from natural-language instructions.

This package implements a compiler-style pipeline:

    Natural language → GP DSL → Validation → Model Builder → Fit → Validation → Recovery

The DSL (gparchitect.dsl.GPSpec) is the single source of truth for all model
specifications. Natural language is an interface only; it never directly constructs
models.

Typical usage::

    from gparchitect.api import run_gparchitect

    model, log = run_gparchitect(
        dataframe=df,
        instruction="Use a Matern52 kernel with ARD on all inputs.",
        input_columns=["x1", "x2", "x3"],
        output_columns=["y"],
    )
"""

from __future__ import annotations

from gparchitect.api import (
    build_model_from_dsl,
    fit_and_validate,
    prepare_data,
    revise_dsl,
    run_gparchitect,
    summarize_attempts,
    translate_to_dsl,
    validate_dsl,
)
from gparchitect.planning import (
    ArchitectureHandoff,
    PlanningRunResult,
    PriorKnowledgeHandoff,
    run_architect,
    run_architecture_focus,
    run_prior_knowledge,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ArchitectureHandoff",
    "PlanningRunResult",
    "PriorKnowledgeHandoff",
    "build_model_from_dsl",
    "fit_and_validate",
    "prepare_data",
    "revise_dsl",
    "run_architect",
    "run_architecture_focus",
    "run_gparchitect",
    "run_prior_knowledge",
    "summarize_attempts",
    "translate_to_dsl",
    "validate_dsl",
]
