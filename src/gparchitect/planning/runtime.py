"""
Executable planning runtime for GPArchitect.

Purpose:
    Implements deterministic planning functions that mirror the prior-knowledge,
    architecture-focus, and planning-orchestrator agent roles in executable Python.

Role in pipeline:
    This subsystem sits upstream of GP DSL translation and produces planning
    artifacts only. It never constructs, fits, or revises GP models.

Inputs / Outputs:
    Inputs are raw planning text or prior-knowledge handoffs.
    Outputs are structured planning models defined in gparchitect.planning.models.

Non-obvious design decisions:
    - The runtime uses explicit keyword-based heuristics instead of an opaque agent
      framework so behavior remains deterministic and inspectable.
    - Routing is exposed as code so the workspace custom agents can delegate to a
      CLI or Python bridge without re-implementing policy.
    - Architecture planning can accept either a parsed prior-knowledge handoff or a
      minimally structured text summary, but it still stops before the DSL.

What this module does NOT do:
    - It does not call translator, builders, fitting, or revision modules.
    - It does not directly construct models from raw text.
    - It does not depend on workspace agent markdown files at runtime.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Literal

from gparchitect.planning.handoff import extract_prior_knowledge_handoff_block, parse_prior_knowledge_handoff_text
from gparchitect.planning.models import (
    ArchitectureHandoff,
    ChosenPath,
    ExtractedKnowledgeItem,
    PlanningMode,
    PlanningRunResult,
    PlanningSourceKind,
    PriorKnowledgeHandoff,
)

logger = logging.getLogger(__name__)


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "structural_behaviors": (
        "smooth",
        "nonlinear",
        "interaction",
        "interact",
        "threshold",
        "stationary",
        "nonstationary",
        "periodic",
        "seasonal",
        "trend",
        "regime",
        "drift",
        "multiscale",
    ),
    "noise_and_uncertainty": (
        "noise",
        "heteroskedastic",
        "uncertain",
        "uncertainty",
        "outlier",
        "heavy tail",
        "measurement error",
        "missing",
        "delayed label",
        "upper",
    ),
    "constraints_and_invariants": (
        "must",
        "should",
        "bounded",
        "positive",
        "monotonic",
        "constraint",
        "invariant",
        "feasible",
        "cannot",
        "never",
    ),
    "feature_grouping_signals": (
        "interact",
        "interaction",
        "couple",
        "together",
        "group",
        "temperature",
        "pressure",
        "cycle count",
        "credit spread",
        "vix",
    ),
    "temporal_or_multiscale_signals": (
        "time",
        "lag",
        "delayed",
        "weekly",
        "monthly",
        "yearly",
        "seasonal",
        "drift",
        "trend",
        "recurrence",
        "memory",
    ),
    "regimes_and_edge_cases": (
        "regime",
        "rare",
        "extreme",
        "edge",
        "sparse",
        "crisis",
        "limit",
        "upper pressure",
    ),
    "data_process_risks": (
        "sampling",
        "label",
        "delayed",
        "leakage",
        "revision",
        "censor",
        "truncation",
        "sensor",
        "instrument",
        "collection",
    ),
    "decision_context": (
        "operator",
        "decision",
        "cost",
        "planning",
        "forecast",
        "yield",
        "degradation",
        "optimization",
    ),
}

_ELICITATION_ONLY_PATTERNS = (
    "elicitation only",
    "only want prior-knowledge elicitation",
    "do not do architecture planning yet",
    "stop after prior knowledge",
    "prior knowledge only",
)


def run_prior_knowledge(
    input_text: str,
    *,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
) -> PriorKnowledgeHandoff:
    """Extract a structured prior-knowledge handoff from planning input.

    Args:
        input_text: Raw prior-knowledge text or an existing handoff block.
        metadata: Optional machine-readable metadata to attach to the result.

    Returns:
        A structured prior-knowledge handoff.
    """

    _raise_if_malformed_prior_handoff(input_text)

    existing_handoff = extract_prior_knowledge_handoff_block(input_text)
    if existing_handoff is not None:
        handoff = parse_prior_knowledge_handoff_text(existing_handoff)
        handoff.metadata.update(metadata or {})
        handoff.metadata.setdefault("source_kind", "prior_knowledge_handoff")
        return handoff

    sentences = _split_sentences(input_text)
    categorized = {key: [] for key in _CATEGORY_KEYWORDS}
    for sentence in sentences:
        lowered = sentence.lower()
        matched = False
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                categorized[category].append(sentence)
                matched = True
        if not matched:
            categorized["structural_behaviors"].append(sentence)

    extracted_knowledge = [_build_knowledge_item(sentence) for sentence in sentences[:8]]
    inputs, outputs = _infer_inputs_and_outputs(sentences)
    assumptions = _infer_validation_assumptions(sentences)
    open_questions = _build_open_questions(categorized, inputs, outputs)
    extension_requirements = _infer_extension_requirements(categorized)

    handoff = PriorKnowledgeHandoff(
        source_text=input_text.strip(),
        system_summary=_build_system_summary(sentences),
        inputs=inputs,
        outputs=outputs,
        controllable_variables=_infer_controllable_variables(sentences),
        latent_or_unobserved_factors=_infer_latent_factors(sentences),
        extracted_knowledge=extracted_knowledge,
        structural_behaviors=_deduplicate(categorized["structural_behaviors"]),
        noise_and_uncertainty=_deduplicate(categorized["noise_and_uncertainty"]),
        constraints_and_invariants=_deduplicate(categorized["constraints_and_invariants"]),
        feature_grouping_signals=_deduplicate(categorized["feature_grouping_signals"]),
        temporal_or_multiscale_signals=_deduplicate(categorized["temporal_or_multiscale_signals"]),
        regimes_and_edge_cases=_deduplicate(categorized["regimes_and_edge_cases"]),
        data_process_risks=_deduplicate(categorized["data_process_risks"]),
        decision_context=_deduplicate(categorized["decision_context"]),
        representability_direct_gp_planning=_infer_direct_gp_planning(categorized),
        representability_preprocessing_or_evaluation=_infer_preprocessing_concerns(categorized),
        representability_likely_extension=extension_requirements,
        representability_unresolved=open_questions,
        candidate_grouping_implications=_infer_grouping_implications(categorized, inputs),
        candidate_kernel_implications=_infer_kernel_implications(categorized),
        candidate_noise_implications=_infer_noise_implications(categorized),
        candidate_priors_implications=_infer_prior_implications(categorized),
        candidate_multitask_implications=_infer_multitask_implications(sentences, outputs),
        candidate_evaluation_implications=_infer_evaluation_implications(categorized),
        candidate_recovery_implications=_infer_recovery_implications(categorized),
        assumptions_requiring_validation=assumptions,
        minimal_open_questions_for_architecture_focus=open_questions,
        metadata={
            "source_kind": "raw_input",
            "sentence_count": len(sentences),
            **(metadata or {}),
        },
    )

    logger.info("Generated prior-knowledge handoff with %d extracted statements.", len(handoff.extracted_knowledge))
    return handoff


def run_architecture_focus(
    handoff: PriorKnowledgeHandoff | str,
    *,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
) -> ArchitectureHandoff:
    """Convert prior-knowledge input into a structured architecture handoff.

    Args:
        handoff: Structured prior-knowledge handoff or a string containing one.
        metadata: Optional machine-readable metadata to attach to the result.

    Returns:
        A structured architecture handoff.
    """

    prior_handoff, source_kind = _coerce_prior_handoff(handoff)
    feature_groups = _build_feature_groups(prior_handoff)
    group_composition = _build_group_composition(prior_handoff)
    kernel_candidates = _build_kernel_candidates(prior_handoff)
    extension_requirements = _build_extension_requirements(prior_handoff)

    architecture = ArchitectureHandoff(
        source_handoff_text=prior_handoff.to_handoff_text(),
        planning_summary=_build_architecture_summary(prior_handoff),
        model_class_implications=_build_model_class_implications(prior_handoff),
        feature_groups=feature_groups,
        group_composition=group_composition,
        kernel_family_candidates=kernel_candidates,
        ard_implications=_build_ard_implications(prior_handoff),
        noise_implications=_build_architecture_noise_implications(prior_handoff),
        prior_implications=_build_architecture_prior_implications(prior_handoff),
        multitask_implications=_build_architecture_multitask_implications(prior_handoff),
        supported_directly=_build_supported_directly(prior_handoff, feature_groups, kernel_candidates),
        preprocessing_or_evaluation_concerns=_build_preprocessing_or_evaluation_concerns(prior_handoff),
        extension_requirements=extension_requirements,
        validation_risks=_build_validation_risks(prior_handoff),
        recovery_risks=_build_recovery_risks(prior_handoff),
        open_questions=_deduplicate(prior_handoff.minimal_open_questions_for_architecture_focus),
        metadata={
            "source_kind": source_kind,
            **(metadata or {}),
        },
    )

    logger.info("Generated architecture handoff with %d kernel candidates.", len(architecture.kernel_family_candidates))
    return architecture


def run_architect(
    input_text: str,
    mode: PlanningMode = "auto",
    *,
    planning_requested: bool | None = None,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
) -> PlanningRunResult:
    """Route planning input through the executable planning runtime.

    Args:
        input_text: Raw prior knowledge, structured summary, or a prior handoff block.
        mode: Explicit planning mode. `auto` applies routing logic.
        planning_requested: Optional override for whether downstream architecture
            planning is requested when `mode` is `auto`.
        metadata: Optional machine-readable metadata attached to the run result.

    Returns:
        Structured result of the executed planning route.
    """

    selected_path, route_reason, source_kind, resolved_planning_requested = _select_route(
        input_text=input_text,
        mode=mode,
        planning_requested=planning_requested,
    )

    prior_handoff: PriorKnowledgeHandoff | None = None
    architecture_handoff: ArchitectureHandoff | None = None

    if selected_path == "Prior Knowledge only":
        prior_handoff = run_prior_knowledge(input_text, metadata=metadata)
    elif selected_path == "Architecture Focus only":
        architecture_handoff = run_architecture_focus(input_text, metadata=metadata)
    else:
        prior_handoff = run_prior_knowledge(input_text, metadata=metadata)
        architecture_handoff = run_architecture_focus(prior_handoff, metadata=metadata)

    unresolved_gaps = []
    if prior_handoff is not None:
        unresolved_gaps.extend(prior_handoff.minimal_open_questions_for_architecture_focus)
    if architecture_handoff is not None:
        unresolved_gaps.extend(architecture_handoff.open_questions)

    return PlanningRunResult(
        chosen_path=selected_path,
        requested_mode=mode,
        source_kind=source_kind,
        planning_requested=resolved_planning_requested,
        route_reason=route_reason,
        prior_knowledge=prior_handoff,
        architecture=architecture_handoff,
        unresolved_gaps=_deduplicate(unresolved_gaps),
        metadata=metadata or {},
    )


def _select_route(
    *,
    input_text: str,
    mode: PlanningMode,
    planning_requested: bool | None,
) -> tuple[ChosenPath, str, PlanningSourceKind, bool]:
    has_handoff = extract_prior_knowledge_handoff_block(input_text) is not None

    if mode == "prior":
        return "Prior Knowledge only", "Explicit prior-only mode requested.", "raw_input", False

    if mode == "architecture":
        source_kind: PlanningSourceKind = "prior_knowledge_handoff" if has_handoff else "structured_summary"
        return "Architecture Focus only", "Explicit architecture-only mode requested.", source_kind, True

    inferred_planning_requested = _infer_planning_requested(input_text) if planning_requested is None else planning_requested
    if has_handoff:
        return (
            "Architecture Focus only",
            "Detected a prior-knowledge handoff block, so routing directly to architecture planning.",
            "prior_knowledge_handoff",
            True,
        )
    if inferred_planning_requested:
        return (
            "Prior Knowledge -> Architecture Focus",
            "Input is raw prior knowledge and architecture planning is requested or implied.",
            "raw_input",
            True,
        )
    return (
        "Prior Knowledge only",
        "Input is raw prior knowledge and the request is limited to elicitation.",
        "raw_input",
        False,
    )


def _infer_planning_requested(input_text: str) -> bool:
    lowered = input_text.lower()
    return not any(pattern in lowered for pattern in _ELICITATION_ONLY_PATTERNS)


def _coerce_prior_handoff(handoff: PriorKnowledgeHandoff | str) -> tuple[PriorKnowledgeHandoff, PlanningSourceKind]:
    if isinstance(handoff, PriorKnowledgeHandoff):
        return handoff, "prior_knowledge_handoff"

    _raise_if_malformed_prior_handoff(handoff)

    block = extract_prior_knowledge_handoff_block(handoff)
    if block is not None:
        return parse_prior_knowledge_handoff_text(block), "prior_knowledge_handoff"

    return run_prior_knowledge(handoff), "structured_summary"


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"[\n\.!?]+", text)
    sentences = [re.sub(r"\s+", " ", chunk).strip(" -") for chunk in chunks if chunk.strip()]
    return sentences[:12] or [text.strip()]


def _raise_if_malformed_prior_handoff(text: str) -> None:
    has_begin = "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in text
    has_end = "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in text
    if (has_begin or has_end) and extract_prior_knowledge_handoff_block(text) is None:
        raise ValueError("Malformed GPArchitect prior-knowledge handoff block.")


def _build_system_summary(sentences: list[str]) -> list[str]:
    if not sentences:
        return ["Planning input was provided without recognizable sentences."]
    summary = [sentences[0]]
    if len(sentences) > 1:
        summary.append(f"Additional planning signals mention: {sentences[1].lower()}.")
    return summary


def _infer_inputs_and_outputs(sentences: list[str]) -> tuple[list[str], list[str]]:
    joined = " ".join(sentences)
    labeled_inputs = _extract_labeled_items(joined, "inputs")
    labeled_outputs = _extract_labeled_items(joined, "outputs")

    if not labeled_inputs:
        interaction_match = re.search(r"([A-Za-z][A-Za-z0-9_ ]+) and ([A-Za-z][A-Za-z0-9_ ]+) interact", joined, re.IGNORECASE)
        if interaction_match:
            labeled_inputs = [interaction_match.group(1).strip(), interaction_match.group(2).strip()]

    if not labeled_outputs:
        output_match = re.search(r"modeling ([A-Za-z][A-Za-z0-9_ ]+?)(?: with| where| and|\.|$)", joined, re.IGNORECASE)
        if output_match:
            labeled_outputs = [output_match.group(1).strip()]

    return _deduplicate(labeled_inputs), _deduplicate(labeled_outputs)


def _extract_labeled_items(text: str, label: str) -> list[str]:
    match = re.search(rf"{label}\s*[:=]?\s*([A-Za-z0-9_,/\- ]+)", text, re.IGNORECASE)
    if match is None:
        return []
    return [item.strip() for item in re.split(r",|\band\b|/", match.group(1)) if item.strip()]


def _infer_controllable_variables(sentences: list[str]) -> list[str]:
    matches = []
    for sentence in sentences:
        if "operator" in sentence.lower() or "controllable" in sentence.lower():
            matches.append(sentence)
    return _deduplicate(matches)


def _infer_latent_factors(sentences: list[str]) -> list[str]:
    matches = []
    for sentence in sentences:
        lowered = sentence.lower()
        if "latent" in lowered or "unobserved" in lowered:
            matches.append(sentence)
    return _deduplicate(matches)


def _build_knowledge_item(sentence: str) -> ExtractedKnowledgeItem:
    lowered = sentence.lower()

    if any(keyword in lowered for keyword in ("must", "bounded", "positive", "constraint", "never")):
        classification = "hard constraint"
        evidence_source = "operational rule"
        planning_status = "requires future DSL or validation extension"
        why_it_matters = "Hard constraints need explicit representability checks before planning can rely on them."
    elif any(keyword in lowered for keyword in ("noise", "observed", "measured", "increases", "weekly", "seasonality")):
        classification = "empirical observation"
        evidence_source = "historical data"
        planning_status = "directly useful for GP architecture planning"
        why_it_matters = "Observed structure can inform kernel and noise planning without collapsing into a model."
    elif any(keyword in lowered for keyword in ("delayed", "leakage", "sampling", "instrument")):
        classification = "data collection artifact"
        evidence_source = "historical data"
        planning_status = "useful for preprocessing or feature engineering"
        why_it_matters = "Data-process artifacts should shape preprocessing and evaluation instead of being buried in architecture prose."
    elif any(keyword in lowered for keyword in ("uncertain", "unknown", "may", "might", "occasional")):
        classification = "open uncertainty"
        evidence_source = "assumption only"
        planning_status = "requires validation before use"
        why_it_matters = "Uncertainty should be preserved and tested rather than silently promoted to a design rule."
    else:
        classification = "soft prior belief"
        evidence_source = "expert judgment"
        planning_status = "requires validation before use"
        why_it_matters = "Domain beliefs are useful signals, but they should remain explicit assumptions until validated."

    confidence = _infer_confidence(lowered)
    return ExtractedKnowledgeItem(
        statement=sentence,
        classification=classification,
        confidence=confidence,
        evidence_source=evidence_source,
        planning_status=planning_status,
        why_it_matters=why_it_matters,
    )


def _infer_confidence(lowered_sentence: str) -> Literal["established", "data-supported", "plausible", "anecdotal", "speculative"]:
    if any(keyword in lowered_sentence for keyword in ("must", "always", "never")):
        return "established"
    if any(keyword in lowered_sentence for keyword in ("observed", "measured", "weekly", "increases", "delayed")):
        return "data-supported"
    if any(keyword in lowered_sentence for keyword in ("may", "might", "occasional", "likely")):
        return "plausible"
    if any(keyword in lowered_sentence for keyword in ("suspect", "believe", "think")):
        return "anecdotal"
    return "speculative"


def _infer_validation_assumptions(sentences: list[str]) -> list[str]:
    assumptions = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ("may", "might", "likely", "occasional", "suspect", "uncertain")):
            assumptions.append(sentence)
    if not assumptions:
        assumptions.append("Clarify which structural claims are empirical observations versus domain assumptions.")
    return _deduplicate(assumptions)


def _build_open_questions(
    categorized: dict[str, list[str]],
    inputs: list[str],
    outputs: list[str],
) -> list[str]:
    questions: list[str] = []
    if not inputs or not outputs:
        questions.append("Which observed inputs drive the target outputs, and which variables are controllable versus contextual?")
    if not categorized["noise_and_uncertainty"]:
        questions.append("Is observation noise roughly constant, or does it vary by regime, range, or operating condition?")
    if not categorized["temporal_or_multiscale_signals"]:
        questions.append("Are there lagged, seasonal, drifting, or multiscale effects that should shape planning?")
    if not categorized["constraints_and_invariants"]:
        questions.append("Are there hard feasibility, monotonicity, boundedness, or conservation constraints to preserve?"
        )
    if not categorized["regimes_and_edge_cases"]:
        questions.append("Which rare regimes or edge cases matter most for evaluation and recovery planning?")
    if not categorized["data_process_risks"]:
        questions.append("Are there data collection artifacts such as delayed labels, leakage, revisions, or instrumentation changes?")
    if not categorized["decision_context"]:
        questions.append("How will the model be used, and which errors are most costly in deployment or analysis?")
    return questions[:7]


def _infer_extension_requirements(categorized: dict[str, list[str]]) -> list[str]:
    requirements = []
    for sentence in categorized["noise_and_uncertainty"]:
        lowered = sentence.lower()
        if "heteroskedastic" in lowered or "noise increases" in lowered:
            requirements.append("Region-dependent noise may require future heteroskedastic-noise support beyond the current scalar noise DSL.")
    for sentence in categorized["constraints_and_invariants"]:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in ("monotonic", "bounded", "positive", "feasible")):
            requirements.append("Hard structural constraints should be treated as validation or future DSL-extension work, not as implicit planner defaults.")
    return _deduplicate(requirements)


def _infer_direct_gp_planning(categorized: dict[str, list[str]]) -> list[str]:
    items = []
    if categorized["structural_behaviors"]:
        items.append("Structural behavior signals can inform kernel-family and composition planning.")
    if categorized["feature_grouping_signals"]:
        items.append("Interaction and grouping cues can inform provisional feature-group planning.")
    if categorized["temporal_or_multiscale_signals"]:
        items.append("Temporal and multiscale cues can inform periodic, RQ, or hierarchical planning options.")
    return items or ["No direct GP planning signals were extracted beyond a generic need for structured prior knowledge."]


def _infer_preprocessing_concerns(categorized: dict[str, list[str]]) -> list[str]:
    items = []
    if categorized["data_process_risks"]:
        items.append("Data-process risks should be handled through preprocessing, leakage checks, and evaluation design.")
    if categorized["decision_context"]:
        items.append("Decision context should influence holdout design and error weighting rather than silently altering architecture.")
    return items or ["No preprocessing-specific concerns were extracted from the current prompt."]


def _infer_grouping_implications(categorized: dict[str, list[str]], inputs: list[str]) -> list[str]:
    items = []
    if categorized["feature_grouping_signals"]:
        items.append("Keep explicitly interacting variables in linked feature groups instead of flattening them into one undifferentiated block.")
    if inputs and len(inputs) > 1:
        items.append("Use provisional per-variable groups first, then relax into shared groups only where the prior knowledge explicitly couples them.")
    return items or ["Input grouping is unresolved; start with provisional singleton groups until stronger evidence appears."]


def _infer_kernel_implications(categorized: dict[str, list[str]]) -> list[str]:
    items = []
    text = " ".join(sum(categorized.values(), []))
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("weekly", "seasonal", "periodic")):
        items.append("Periodic structure is plausible for recurring behavior and should stay explicit in architecture planning.")
    if any(keyword in lowered for keyword in ("multiscale", "regime", "drift")):
        items.append("Multiscale or nonstationary signals suggest RQ, Spectral Mixture, or careful hierarchical planning rather than a single smooth kernel assumption.")
    if any(keyword in lowered for keyword in ("smooth", "yield", "degradation")):
        items.append("Smooth baseline structure supports starting with Matern or RBF-style candidates during architecture planning.")
    if any(keyword in lowered for keyword in ("interaction", "interact")):
        items.append("Interactions should be preserved through additive-plus-product or hierarchical composition rather than buried in prose.")
    return items or ["Kernel-family evidence is weak; retain broad candidates until more structure is confirmed."]


def _infer_noise_implications(categorized: dict[str, list[str]]) -> list[str]:
    if categorized["noise_and_uncertainty"]:
        return [
            "Noise behavior should remain explicit in planning, especially when it varies by regime or operating range.",
        ]
    return ["Noise assumptions are currently underspecified and should default to simple scalar-noise planning until clarified."]


def _infer_prior_implications(categorized: dict[str, list[str]]) -> list[str]:
    if categorized["constraints_and_invariants"]:
        return [
            "Only add explicit priors if the user can quantify them; otherwise preserve constraints as planning notes and validation risks.",
        ]
    return ["Use BoTorch defaults unless the prior knowledge becomes quantified enough to justify explicit priors."]


def _infer_multitask_implications(sentences: list[str], outputs: list[str]) -> list[str]:
    joined = " ".join(sentences).lower()
    if "task" in joined or len(outputs) > 1:
        return ["Multiple outputs or task indicators may justify MultiTaskGP or ModelListGP planning."]
    return ["Current prompt looks compatible with single-output planning unless later information introduces task structure."]


def _infer_evaluation_implications(categorized: dict[str, list[str]]) -> list[str]:
    items = []
    if categorized["regimes_and_edge_cases"]:
        items.append("Evaluation should isolate rare regimes and edge cases rather than averaging them away.")
    if categorized["data_process_risks"]:
        items.append("Evaluation should explicitly guard against leakage, delayed labels, or instrumentation changes.")
    return items or ["Define evaluation splits that preserve the deployment context before refining architecture choices."]


def _infer_recovery_implications(categorized: dict[str, list[str]]) -> list[str]:
    items = []
    if categorized["structural_behaviors"]:
        items.append("Recovery should be prepared to simplify overly rich compositions if fitting becomes unstable.")
    if categorized["noise_and_uncertainty"]:
        items.append("Recovery may need to fall back to simpler scalar-noise assumptions when richer noise stories are not directly representable.")
    return items or ["Recovery planning should preserve a path back to simpler kernels and default noise assumptions."]


def _build_feature_groups(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    feature_groups: list[str] = []
    if prior_handoff.feature_grouping_signals:
        feature_groups.extend(prior_handoff.feature_grouping_signals)
    elif prior_handoff.inputs:
        feature_groups.extend(f"Provisional singleton group for {feature_name}." for feature_name in prior_handoff.inputs)
    else:
        feature_groups.append("Inputs are not enumerated yet, so feature groups remain provisional.")
    return _deduplicate(feature_groups)


def _build_group_composition(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    lowered = " ".join(prior_handoff.feature_grouping_signals + prior_handoff.structural_behaviors).lower()
    if "interact" in lowered or "interaction" in lowered:
        return ["Prefer hierarchical composition so main effects remain explicit while preserving interaction terms."]
    return ["Prefer additive composition unless later evidence justifies multiplicative or hierarchical structure."]


def _build_kernel_candidates(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    items: list[str] = []
    combined = " ".join(
        prior_handoff.structural_behaviors + prior_handoff.temporal_or_multiscale_signals + prior_handoff.regimes_and_edge_cases
    ).lower()
    if any(keyword in combined for keyword in ("weekly", "seasonal", "periodic")):
        items.append("Periodic kernel candidate for recurring structure.")
    if any(keyword in combined for keyword in ("multiscale", "regime", "drift")):
        items.append("Rational Quadratic or Spectral Mixture candidate for multi-scale or regime-sensitive structure.")
    if any(keyword in combined for keyword in ("interaction", "interact")):
        items.append("Composable Matern-family kernels with hierarchical products to preserve interactions.")
    if any(keyword in combined for keyword in ("smooth", "yield", "degradation")):
        items.append("Matern52 or RBF candidate as a smooth baseline.")
    if not items:
        items.append("Matern52 baseline candidate until stronger structure is established.")
    return _deduplicate(items)


def _build_architecture_summary(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    summary = []
    if prior_handoff.system_summary:
        summary.extend(prior_handoff.system_summary[:2])
    if prior_handoff.noise_and_uncertainty:
        summary.append("Noise behavior is planning-relevant and should remain explicit in the DSL-facing handoff.")
    if prior_handoff.data_process_risks:
        summary.append("Data-process risks are important enough to separate from architecture and handle in preprocessing/evaluation.")
    return _deduplicate(summary)


def _build_model_class_implications(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    if len(prior_handoff.outputs) > 1:
        return ["Multiple outputs suggest ModelListGP or MultiTaskGP planning depending on expected cross-output correlation."]
    if any("task" in signal.lower() for signal in prior_handoff.candidate_multitask_implications):
        return ["Task structure should be evaluated explicitly before choosing between SingleTaskGP and MultiTaskGP."]
    return ["SingleTaskGP is the default planning baseline unless task or multi-output structure becomes explicit."]


def _build_ard_implications(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    if len(prior_handoff.inputs) > 1 or len(prior_handoff.feature_grouping_signals) > 1:
        return ["ARD is likely useful because multiple inputs or groups may vary in relevance."]
    return ["ARD is optional; revisit once input dimensionality and grouping are confirmed."]


def _build_architecture_noise_implications(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    lowered = " ".join(prior_handoff.noise_and_uncertainty).lower()
    if "heteroskedastic" in lowered or "noise increases" in lowered:
        return ["Current DSL supports scalar noise planning; region-dependent noise should be flagged as an extension risk."]
    if prior_handoff.noise_and_uncertainty:
        return ["Preserve explicit noise assumptions in planning, but keep them separate from kernel selection."]
    return ["Noise assumptions are underspecified, so architecture should start from learnable scalar noise."]


def _build_architecture_prior_implications(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    if prior_handoff.constraints_and_invariants:
        return ["Do not encode hard constraints as priors unless they can be expressed quantitatively and validated downstream."]
    return ["Use default priors unless domain knowledge becomes quantitative enough to justify explicit specification."]


def _build_architecture_multitask_implications(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    return _deduplicate(prior_handoff.candidate_multitask_implications)


def _build_supported_directly(
    prior_handoff: PriorKnowledgeHandoff,
    feature_groups: list[str],
    kernel_candidates: list[str],
) -> list[str]:
    supported = []
    if feature_groups:
        supported.append("Feature groups can be represented directly in the current DSL.")
    if kernel_candidates:
        supported.append("Kernel-family candidates can be expressed as DSL-level planning choices without constructing a model.")
    if prior_handoff.noise_and_uncertainty:
        supported.append("Simple scalar-noise assumptions map cleanly into the current DSL.")
    return _deduplicate(supported)


def _build_preprocessing_or_evaluation_concerns(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    concerns = []
    if prior_handoff.data_process_risks:
        concerns.extend(prior_handoff.data_process_risks)
    concerns.extend(prior_handoff.candidate_evaluation_implications)
    return _deduplicate(concerns) or ["No explicit preprocessing or evaluation concerns were extracted."]


def _build_extension_requirements(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    requirements = []
    requirements.extend(prior_handoff.representability_likely_extension)
    if any(term in " ".join(prior_handoff.constraints_and_invariants).lower() for term in ("monotonic", "bounded", "positive")):
        requirements.append("Strict functional constraints likely need future validation or DSL extensions.")
    return _deduplicate(requirements)


def _build_validation_risks(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    risks = []
    if not prior_handoff.inputs:
        risks.append("Input feature identities are incomplete, which weakens feature-group validation.")
    if prior_handoff.representability_likely_extension:
        risks.append("Some requested behaviors exceed current DSL support and must stay explicit as risks.")
    if prior_handoff.assumptions_requiring_validation:
        risks.append("Several planning signals remain assumptions and should not be treated as validated structure.")
    return _deduplicate(risks)


def _build_recovery_risks(prior_handoff: PriorKnowledgeHandoff) -> list[str]:
    risks = []
    if prior_handoff.feature_grouping_signals:
        risks.append("Rich interaction structure can lead to over-complex initial plans that later need simplification.")
    if prior_handoff.noise_and_uncertainty:
        risks.append("Unresolved noise stories can force fallback to simpler scalar-noise assumptions during recovery.")
    if prior_handoff.regimes_and_edge_cases:
        risks.append("Sparse regimes may leave the eventual model vulnerable to brittle extrapolation and revision churn.")
    return _deduplicate(risks)


def _deduplicate(items: list[str]) -> list[str]:
    ordered = OrderedDict[str, None]()
    for item in items:
        clean = item.strip()
        if clean:
            ordered.setdefault(clean, None)
    return list(ordered.keys())
