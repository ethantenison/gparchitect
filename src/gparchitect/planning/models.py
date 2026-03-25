"""
Structured planning artifacts for GPArchitect.

Purpose:
    Defines typed, JSON-serializable planning models for the executable planning
    subsystem that sits upstream of GP DSL construction.

Role in pipeline:
    Natural language planning input -> planning handoffs -> downstream DSL work.
    These models intentionally stop before translation into the GP DSL.

Inputs / Outputs:
    Inputs are plain Python values used to instantiate Pydantic models.
    Outputs are structured planning artifacts that can be serialized to JSON or
    rendered as explicit handoff blocks for humans and tooling.

Non-obvious design decisions:
    - Handoff text is derived from structured fields rather than treated as the
      primary source of truth.
    - The display-oriented handoff block format is stable and parseable.
    - The models are intentionally explicit about unresolved gaps so planning can
      remain inspectable and deterministic.

What this module does NOT do:
    - It does not perform routing or inference from raw text.
    - It does not translate planning output into the GP DSL.
    - It does not construct or fit models.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

PlanningMode: TypeAlias = Literal["auto", "prior", "architecture"]
ChosenPath: TypeAlias = Literal[
    "Prior Knowledge only",
    "Architecture Focus only",
    "Prior Knowledge -> Architecture Focus",
]
PlanningSourceKind: TypeAlias = Literal["raw_input", "prior_knowledge_handoff", "structured_summary"]


def _render_list(items: list[str], *, empty_text: str = "- none") -> str:
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def _render_named_value(label: str, items: list[str]) -> str:
    if not items:
        return f"- {label}: none"
    return f"- {label}: {', '.join(items)}"


class ExtractedKnowledgeItem(BaseModel):
    """One planning-relevant statement extracted from raw prior knowledge."""

    statement: str
    classification: Literal[
        "hard constraint",
        "soft prior belief",
        "empirical observation",
        "data collection artifact",
        "implementation requirement",
        "open uncertainty",
    ]
    confidence: Literal["established", "data-supported", "plausible", "anecdotal", "speculative"]
    evidence_source: Literal[
        "mechanism",
        "expert judgment",
        "historical data",
        "prior experiment",
        "operational rule",
        "regulatory rule",
        "assumption only",
    ]
    planning_status: Literal[
        "directly useful for GP architecture planning",
        "useful for preprocessing or feature engineering",
        "useful for evaluation design",
        "requires validation before use",
        "requires future DSL or validation extension",
        "not yet actionable",
    ]
    why_it_matters: str

    def to_handoff_lines(self) -> list[str]:
        """Render the knowledge item using the handoff block format."""

        return [
            f"- Statement: {self.statement}",
            f"  Classification: {self.classification}",
            f"  Confidence: {self.confidence}",
            f"  Evidence source: {self.evidence_source}",
            f"  Planning status: {self.planning_status}",
            f"  Why it matters: {self.why_it_matters}",
        ]


class PriorKnowledgeHandoff(BaseModel):
    """Structured output of the executable prior-knowledge stage."""

    source_text: str
    system_summary: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    controllable_variables: list[str] = Field(default_factory=list)
    latent_or_unobserved_factors: list[str] = Field(default_factory=list)
    extracted_knowledge: list[ExtractedKnowledgeItem] = Field(default_factory=list)
    structural_behaviors: list[str] = Field(default_factory=list)
    noise_and_uncertainty: list[str] = Field(default_factory=list)
    constraints_and_invariants: list[str] = Field(default_factory=list)
    feature_grouping_signals: list[str] = Field(default_factory=list)
    temporal_or_multiscale_signals: list[str] = Field(default_factory=list)
    regimes_and_edge_cases: list[str] = Field(default_factory=list)
    data_process_risks: list[str] = Field(default_factory=list)
    decision_context: list[str] = Field(default_factory=list)
    representability_direct_gp_planning: list[str] = Field(default_factory=list)
    representability_preprocessing_or_evaluation: list[str] = Field(default_factory=list)
    representability_likely_extension: list[str] = Field(default_factory=list)
    representability_unresolved: list[str] = Field(default_factory=list)
    candidate_grouping_implications: list[str] = Field(default_factory=list)
    candidate_kernel_implications: list[str] = Field(default_factory=list)
    candidate_noise_implications: list[str] = Field(default_factory=list)
    candidate_priors_implications: list[str] = Field(default_factory=list)
    candidate_multitask_implications: list[str] = Field(default_factory=list)
    candidate_evaluation_implications: list[str] = Field(default_factory=list)
    candidate_recovery_implications: list[str] = Field(default_factory=list)
    assumptions_requiring_validation: list[str] = Field(default_factory=list)
    minimal_open_questions_for_architecture_focus: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)

    def to_handoff_text(self) -> str:
        """Return the canonical prior-knowledge handoff block."""

        knowledge_lines: list[str] = []
        for item in self.extracted_knowledge:
            knowledge_lines.extend(item.to_handoff_lines())

        extracted_knowledge_text = "\n".join(knowledge_lines) if knowledge_lines else "- none"

        return "\n".join(
            [
                "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF",
                "",
                "System Summary:",
                _render_list(self.system_summary),
                "",
                "Inputs And Outputs:",
                _render_named_value("Inputs", self.inputs),
                _render_named_value("Outputs", self.outputs),
                _render_named_value("Controllable variables", self.controllable_variables),
                _render_named_value("Latent or unobserved factors", self.latent_or_unobserved_factors),
                "",
                "Extracted Knowledge:",
                extracted_knowledge_text,
                "",
                "Structural Behaviors:",
                _render_list(self.structural_behaviors),
                "",
                "Noise And Uncertainty:",
                _render_list(self.noise_and_uncertainty),
                "",
                "Constraints And Invariants:",
                _render_list(self.constraints_and_invariants),
                "",
                "Feature Grouping Signals:",
                _render_list(self.feature_grouping_signals),
                "",
                "Temporal Or Multiscale Signals:",
                _render_list(self.temporal_or_multiscale_signals),
                "",
                "Regimes And Edge Cases:",
                _render_list(self.regimes_and_edge_cases),
                "",
                "Data Process Risks:",
                _render_list(self.data_process_risks),
                "",
                "Decision Context:",
                _render_list(self.decision_context),
                "",
                "Representability Assessment:",
                _render_named_value("Direct GP planning", self.representability_direct_gp_planning),
                _render_named_value("Preprocessing or evaluation", self.representability_preprocessing_or_evaluation),
                _render_named_value(
                    "Likely future DSL or validation extension",
                    self.representability_likely_extension,
                ),
                _render_named_value("Unresolved or not actionable", self.representability_unresolved),
                "",
                "Architecture-Relevant Signals:",
                _render_named_value("Candidate grouping implications", self.candidate_grouping_implications),
                _render_named_value("Candidate kernel implications", self.candidate_kernel_implications),
                _render_named_value("Candidate noise implications", self.candidate_noise_implications),
                _render_named_value("Candidate priors implications", self.candidate_priors_implications),
                _render_named_value("Candidate multitask implications", self.candidate_multitask_implications),
                _render_named_value("Candidate evaluation implications", self.candidate_evaluation_implications),
                _render_named_value("Candidate recovery implications", self.candidate_recovery_implications),
                "",
                "Assumptions Requiring Validation:",
                _render_list(self.assumptions_requiring_validation),
                "",
                "Minimal Open Questions For Architecture Focus:",
                _render_list(self.minimal_open_questions_for_architecture_focus),
                "",
                "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF",
            ]
        )


class ArchitectureHandoff(BaseModel):
    """Structured output of the executable architecture-planning stage."""

    source_handoff_text: str
    planning_summary: list[str] = Field(default_factory=list)
    model_class_implications: list[str] = Field(default_factory=list)
    feature_groups: list[str] = Field(default_factory=list)
    group_composition: list[str] = Field(default_factory=list)
    kernel_family_candidates: list[str] = Field(default_factory=list)
    ard_implications: list[str] = Field(default_factory=list)
    noise_implications: list[str] = Field(default_factory=list)
    prior_implications: list[str] = Field(default_factory=list)
    multitask_implications: list[str] = Field(default_factory=list)
    supported_directly: list[str] = Field(default_factory=list)
    preprocessing_or_evaluation_concerns: list[str] = Field(default_factory=list)
    extension_requirements: list[str] = Field(default_factory=list)
    validation_risks: list[str] = Field(default_factory=list)
    recovery_risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)

    def to_handoff_text(self) -> str:
        """Return the canonical architecture handoff block."""

        return "\n".join(
            [
                "BEGIN GPARCHITECT ARCHITECTURE HANDOFF",
                "",
                "Planning Summary:",
                _render_list(self.planning_summary),
                "",
                "Candidate DSL-Level Decisions:",
                _render_named_value("Model class implications", self.model_class_implications),
                _render_named_value("Feature groups", self.feature_groups),
                _render_named_value("Group composition", self.group_composition),
                _render_named_value("Kernel-family candidates", self.kernel_family_candidates),
                _render_named_value("ARD implications", self.ard_implications),
                _render_named_value("Noise implications", self.noise_implications),
                _render_named_value("Prior implications", self.prior_implications),
                _render_named_value("Multitask implications", self.multitask_implications),
                "",
                "Representability:",
                _render_named_value("Supported directly", self.supported_directly),
                _render_named_value(
                    "Likely preprocessing or evaluation concerns",
                    self.preprocessing_or_evaluation_concerns,
                ),
                _render_named_value("Likely extension requirements", self.extension_requirements),
                "",
                "Validation Risks:",
                _render_list(self.validation_risks),
                "",
                "Recovery Risks:",
                _render_list(self.recovery_risks),
                "",
                "Open Questions:",
                _render_list(self.open_questions),
                "",
                "END GPARCHITECT ARCHITECTURE HANDOFF",
            ]
        )


class PlanningRunResult(BaseModel):
    """Combined result of the executable planning runtime."""

    chosen_path: ChosenPath
    requested_mode: PlanningMode
    source_kind: PlanningSourceKind
    planning_requested: bool
    route_reason: str
    prior_knowledge: PriorKnowledgeHandoff | None = None
    architecture: ArchitectureHandoff | None = None
    unresolved_gaps: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Render a concise human-readable planning result."""

        sections = [
            "Chosen Path:",
            f"- {self.chosen_path}",
            "",
            "Why:",
            f"- {self.route_reason}",
        ]

        if self.prior_knowledge is not None:
            sections.extend(["", self.prior_knowledge.to_handoff_text()])

        if self.architecture is not None:
            sections.extend(["", self.architecture.to_handoff_text()])

        sections.extend(["", "Remaining Gaps:", _render_list(self.unresolved_gaps)])
        return "\n".join(sections)
