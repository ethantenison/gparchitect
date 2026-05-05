"""Legacy KernelSpec normalization for GPArchitect.

Purpose:
    Provides adapter functions to convert old KernelSpec-shaped payloads
    into the new canonical KernelExpr discriminated union.

Role in pipeline:
    Called during DSL deserialization when loading old payloads.
    Not part of the main NL → DSL → ... pipeline for new specs.

What this module does NOT do:
    - It does not silently reinterpret ambiguous inputs.
    - It does not validate the resulting KernelExpr (see validation module).
"""

from __future__ import annotations

from gparchitect.dsl.schema import (
    KernelExpr,
    KernelSpec,
    _kernel_spec_to_expr,
)


def normalize_kernel_spec(spec: KernelSpec) -> KernelExpr:  # type: ignore[type-arg]
    """Convert a legacy KernelSpec to the canonical KernelExpr discriminated union.

    Dispatches by the shape of the old spec:
    - kernel_type == CHANGEPOINT → ChangepointKernelSpec (children[0] = before, children[1] = after)
    - children non-empty with ADDITIVE or MULTIPLICATIVE → CompositeKernelSpec
    - otherwise → LeafKernelSpec
    - children non-empty with composition=NONE → raises ValueError

    Args:
        spec: A legacy KernelSpec instance.

    Returns:
        A LeafKernelSpec, CompositeKernelSpec, or ChangepointKernelSpec.

    Raises:
        ValueError: If the spec has an ambiguous or invalid shape.
    """
    return _kernel_spec_to_expr(spec)
