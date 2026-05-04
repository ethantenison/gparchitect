"""
Parsing helpers for GPArchitect planning handoff blocks.

Purpose:
    Provides deterministic parsing and validation for the text handoff formats
    used by the planning subsystem and workspace custom agents.

Role in pipeline:
    Bridges machine-readable planning models and the stable handoff block text
    exchanged between planning stages.

Inputs / Outputs:
    Accepts raw handoff text and returns structured Pydantic planning models.

Non-obvious design decisions:
    - Parsing is intentionally strict about begin/end markers so malformed inputs
      fail loudly instead of silently degrading.
    - The parser is optimized for the handoff format emitted by this package,
      while remaining permissive enough for lightly edited user text.

What this module does NOT do:
    - It does not infer planning content from raw prose.
    - It does not route between planning stages.
    - It does not construct GP DSL objects.
"""

from __future__ import annotations

import re
from collections import defaultdict

from gparchitect.planning.models import (
    ArchitectureHandoff,
    ExtractedKnowledgeItem,
    PriorKnowledgeHandoff,
)

PRIOR_BEGIN = "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF"
PRIOR_END = "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF"
ARCHITECTURE_BEGIN = "BEGIN GPARCHITECT ARCHITECTURE HANDOFF"
ARCHITECTURE_END = "END GPARCHITECT ARCHITECTURE HANDOFF"


def extract_prior_knowledge_handoff_block(text: str) -> str | None:
    """Return the prior-knowledge handoff block if present."""

    return _extract_block(text, PRIOR_BEGIN, PRIOR_END)


def extract_architecture_handoff_block(text: str) -> str | None:
    """Return the architecture handoff block if present."""

    return _extract_block(text, ARCHITECTURE_BEGIN, ARCHITECTURE_END)


def parse_prior_knowledge_handoff_text(text: str) -> PriorKnowledgeHandoff:
    """Parse a prior-knowledge handoff block into a structured model.

    Args:
        text: String containing the canonical handoff block.

    Returns:
        A structured prior-knowledge handoff.

    Raises:
        ValueError: If the block is missing or malformed.
    """

    block = extract_prior_knowledge_handoff_block(text)
    if block is None:
        raise ValueError("Input does not contain a valid GPArchitect prior-knowledge handoff block.")

    section_map = _parse_sections(block, PRIOR_BEGIN, PRIOR_END)
    inputs_and_outputs = _parse_named_list_section(section_map.get("Inputs And Outputs", []))
    representability = _parse_named_list_section(section_map.get("Representability Assessment", []))
    architecture_signals = _parse_named_list_section(section_map.get("Architecture-Relevant Signals", []))

    return PriorKnowledgeHandoff(
        source_text=block,
        system_summary=_strip_bullets(section_map.get("System Summary", [])),
        inputs=_split_value(inputs_and_outputs.get("Inputs", "none")),
        outputs=_split_value(inputs_and_outputs.get("Outputs", "none")),
        controllable_variables=_split_value(inputs_and_outputs.get("Controllable variables", "none")),
        latent_or_unobserved_factors=_split_value(inputs_and_outputs.get("Latent or unobserved factors", "none")),
        extracted_knowledge=_parse_extracted_knowledge(section_map.get("Extracted Knowledge", [])),
        structural_behaviors=_strip_bullets(section_map.get("Structural Behaviors", [])),
        noise_and_uncertainty=_strip_bullets(section_map.get("Noise And Uncertainty", [])),
        constraints_and_invariants=_strip_bullets(section_map.get("Constraints And Invariants", [])),
        feature_grouping_signals=_strip_bullets(section_map.get("Feature Grouping Signals", [])),
        temporal_or_multiscale_signals=_strip_bullets(section_map.get("Temporal Or Multiscale Signals", [])),
        regimes_and_edge_cases=_strip_bullets(section_map.get("Regimes And Edge Cases", [])),
        data_process_risks=_strip_bullets(section_map.get("Data Process Risks", [])),
        decision_context=_strip_bullets(section_map.get("Decision Context", [])),
        representability_direct_gp_planning=_split_value(representability.get("Direct GP planning", "none")),
        representability_preprocessing_or_evaluation=_split_value(
            representability.get("Preprocessing or evaluation", "none")
        ),
        representability_likely_extension=_split_value(
            representability.get("Likely future DSL or validation extension", "none")
        ),
        representability_unresolved=_split_value(representability.get("Unresolved or not actionable", "none")),
        candidate_grouping_implications=_split_value(
            architecture_signals.get("Candidate grouping implications", "none")
        ),
        candidate_kernel_implications=_split_value(architecture_signals.get("Candidate kernel implications", "none")),
        candidate_noise_implications=_split_value(architecture_signals.get("Candidate noise implications", "none")),
        candidate_priors_implications=_split_value(architecture_signals.get("Candidate priors implications", "none")),
        candidate_multitask_implications=_split_value(
            architecture_signals.get("Candidate multitask implications", "none")
        ),
        candidate_evaluation_implications=_split_value(
            architecture_signals.get("Candidate evaluation implications", "none")
        ),
        candidate_recovery_implications=_split_value(
            architecture_signals.get("Candidate recovery implications", "none")
        ),
        assumptions_requiring_validation=_strip_bullets(section_map.get("Assumptions Requiring Validation", [])),
        minimal_open_questions_for_architecture_focus=_strip_bullets(
            section_map.get("Minimal Open Questions For Architecture Focus", [])
        ),
    )


def parse_architecture_handoff_text(text: str) -> ArchitectureHandoff:
    """Parse an architecture handoff block into a structured model."""

    block = extract_architecture_handoff_block(text)
    if block is None:
        raise ValueError("Input does not contain a valid GPArchitect architecture handoff block.")

    section_map = _parse_sections(block, ARCHITECTURE_BEGIN, ARCHITECTURE_END)
    candidate_decisions = _parse_named_list_section(section_map.get("Candidate DSL-Level Decisions", []))
    representability = _parse_named_list_section(section_map.get("Representability", []))

    return ArchitectureHandoff(
        source_handoff_text=block,
        planning_summary=_strip_bullets(section_map.get("Planning Summary", [])),
        model_class_implications=_split_value(candidate_decisions.get("Model class implications", "none")),
        feature_groups=_split_value(candidate_decisions.get("Feature groups", "none")),
        group_composition=_split_value(candidate_decisions.get("Group composition", "none")),
        kernel_family_candidates=_split_value(candidate_decisions.get("Kernel-family candidates", "none")),
        ard_implications=_split_value(candidate_decisions.get("ARD implications", "none")),
        noise_implications=_split_value(candidate_decisions.get("Noise implications", "none")),
        prior_implications=_split_value(candidate_decisions.get("Prior implications", "none")),
        multitask_implications=_split_value(candidate_decisions.get("Multitask implications", "none")),
        supported_directly=_split_value(representability.get("Supported directly", "none")),
        preprocessing_or_evaluation_concerns=_split_value(
            representability.get("Likely preprocessing or evaluation concerns", "none")
        ),
        extension_requirements=_split_value(representability.get("Likely extension requirements", "none")),
        validation_risks=_strip_bullets(section_map.get("Validation Risks", [])),
        recovery_risks=_strip_bullets(section_map.get("Recovery Risks", [])),
        open_questions=_strip_bullets(section_map.get("Open Questions", [])),
    )


def _extract_block(text: str, begin_marker: str, end_marker: str) -> str | None:
    begin_index = text.find(begin_marker)
    if begin_index == -1:
        return None
    end_index = text.find(end_marker, begin_index)
    if end_index == -1 or end_index < begin_index:
        return None
    end_index += len(end_marker)
    return text[begin_index:end_index].strip()


def _parse_sections(block: str, begin_marker: str, end_marker: str) -> dict[str, list[str]]:
    lines = [line.rstrip() for line in block.splitlines()]
    if not lines or lines[0] != begin_marker or lines[-1] != end_marker:
        raise ValueError("Malformed handoff block markers.")

    section_map: dict[str, list[str]] = defaultdict(list)
    current_section: str | None = None

    for line in lines[1:-1]:
        if not line.strip():
            continue
        if not line.startswith("-") and line.endswith(":"):
            current_section = line[:-1]
            continue
        if current_section is None:
            raise ValueError("Encountered handoff content outside a named section.")
        section_map[current_section].append(line)

    return dict(section_map)


def _parse_named_list_section(lines: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in lines:
        clean = line.lstrip("- ").strip()
        if ":" not in clean:
            continue
        key, value = clean.split(":", maxsplit=1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_extracted_knowledge(lines: list[str]) -> list[ExtractedKnowledgeItem]:
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("- Statement:"):
            if current is not None:
                items.append(current)
            _, value = line.split(":", maxsplit=1)
            current = {"Statement": value.strip()}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        current[key.strip()] = value.strip()

    if current is not None:
        items.append(current)

    parsed_items: list[ExtractedKnowledgeItem] = []
    for item in items:
        parsed_items.append(
            ExtractedKnowledgeItem(
                statement=item.get("Statement", "none"),
                classification=item.get("Classification", "soft prior belief"),
                confidence=item.get("Confidence", "plausible"),
                evidence_source=item.get("Evidence source", "assumption only"),
                planning_status=item.get("Planning status", "requires validation before use"),
                why_it_matters=item.get("Why it matters", "Provides a planning signal that should be inspected."),
            )
        )
    return parsed_items


def _strip_bullets(lines: list[str]) -> list[str]:
    values = [line.lstrip("- ").strip() for line in lines if line.strip()]
    return [] if values == ["none"] else values


def _split_value(value: str) -> list[str]:
    clean = value.strip()
    if not clean or clean.lower() == "none":
        return []
    parts = [segment.strip() for segment in re.split(r";|,|\band\b", clean) if segment.strip()]
    return parts or [clean]
