"""
Executable planning subsystem for GPArchitect.

Purpose:
    Exposes deterministic planning APIs and structured handoff models that sit
    upstream of the GP DSL pipeline.

Role in pipeline:
    Planning is optional pre-translation work that helps organize prior knowledge
    and architecture intent before any DSL is produced.

Inputs / Outputs:
    Accepts raw planning text or structured handoffs and returns typed planning
    artifacts suitable for JSON serialization or CLI output.

Non-obvious design decisions:
    - The planning runtime is separate from gparchitect.api so the model-building
      pipeline boundary remains explicit.
    - Handoff parsing helpers are exported because the workspace custom agents and
      tests need a stable bridge format.

What this module does NOT do:
    - It does not build, fit, or revise models.
    - It does not translate planning artifacts into the GP DSL.
"""

from __future__ import annotations

from gparchitect.planning.handoff import (
    extract_architecture_handoff_block,
    extract_prior_knowledge_handoff_block,
    parse_architecture_handoff_text,
    parse_prior_knowledge_handoff_text,
)
from gparchitect.planning.models import (
    ArchitectureHandoff,
    PlanningRunResult,
    PriorKnowledgeHandoff,
)
from gparchitect.planning.runtime import run_architect, run_architecture_focus, run_prior_knowledge

__all__ = [
    "ArchitectureHandoff",
    "PlanningRunResult",
    "PriorKnowledgeHandoff",
    "extract_architecture_handoff_block",
    "extract_prior_knowledge_handoff_block",
    "parse_architecture_handoff_text",
    "parse_prior_knowledge_handoff_text",
    "run_architect",
    "run_architecture_focus",
    "run_prior_knowledge",
]
