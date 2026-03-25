"""Tests for the executable GPArchitect planning runtime."""

from __future__ import annotations

import json

import pytest
from gparchitect import run_architect, run_architecture_focus, run_prior_knowledge
from gparchitect.planning import parse_prior_knowledge_handoff_text
from gparchitect.planning.models import ExtractedKnowledgeItem, PriorKnowledgeHandoff


class TestPlanningModels:
    """Validate planning-model serialization and handoff formatting."""

    def test_prior_knowledge_handoff_serializes_and_round_trips(self) -> None:
        handoff = PriorKnowledgeHandoff(
            source_text="Battery degradation depends on temperature and cycle count.",
            system_summary=["Battery degradation forecasting with temperature and cycle count inputs."],
            inputs=["temperature", "cycle_count"],
            outputs=["capacity_fade"],
            extracted_knowledge=[
                ExtractedKnowledgeItem(
                    statement="Battery degradation depends on temperature and cycle count.",
                    classification="empirical observation",
                    confidence="data-supported",
                    evidence_source="historical data",
                    planning_status="directly useful for GP architecture planning",
                    why_it_matters="It affects feature grouping and kernel-family planning.",
                )
            ],
            structural_behaviors=["Capacity fade is smooth over most of the operating range."],
            assumptions_requiring_validation=["Confirm whether the smooth trend breaks in fast-charging regimes."],
            minimal_open_questions_for_architecture_focus=["Is there a task indicator for cell chemistry or operating mode?"],
        )

        payload = json.loads(handoff.model_dump_json())
        round_trip = parse_prior_knowledge_handoff_text(handoff.to_handoff_text())

        assert payload["inputs"] == ["temperature", "cycle_count"]
        assert payload["outputs"] == ["capacity_fade"]
        assert round_trip.inputs == ["temperature", "cycle_count"]
        assert round_trip.outputs == ["capacity_fade"]
        assert round_trip.extracted_knowledge[0].classification == "empirical observation"


class TestPlanningRuntime:
    """Validate planning extraction, routing, and architecture mapping."""

    def test_run_prior_knowledge_extracts_grouping_and_noise_signals(self) -> None:
        result = run_prior_knowledge(
            "We are modeling chemical process yield. Temperature and pressure interact. "
            "Noise increases near the upper pressure limit. Operators want downstream planning."
        )

        assert any("temperature" in item.lower() for item in result.feature_grouping_signals)
        assert any("noise increases" in item.lower() for item in result.noise_and_uncertainty)
        assert "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in result.to_handoff_text()

    def test_run_architecture_focus_maps_periodic_and_data_process_signals(self) -> None:
        prior = run_prior_knowledge(
            "Demand forecasting has weekly seasonality. Labels are delayed by one week. "
            "Occasional regime shifts occur around promotions."
        )

        result = run_architecture_focus(prior)

        assert any("Periodic" in item for item in result.kernel_family_candidates)
        assert any("delayed" in item.lower() for item in result.preprocessing_or_evaluation_concerns)
        assert "BEGIN GPARCHITECT ARCHITECTURE HANDOFF" in result.to_handoff_text()

    def test_run_architect_auto_routes_raw_planning_input_through_both_stages(self) -> None:
        result = run_architect(
            "Temperature and pressure interact. Noise increases near the upper pressure limit. "
            "Operators want end-to-end GPArchitect planning from these notes.",
            mode="auto",
        )

        assert result.chosen_path == "Prior Knowledge -> Architecture Focus"
        assert result.prior_knowledge is not None
        assert result.architecture is not None

    def test_run_architect_auto_respects_elicitation_only_language(self) -> None:
        result = run_architect(
            "I only want prior-knowledge elicitation for a demand forecasting system with weekly seasonality, "
            "delayed labels, and occasional regime shifts. Do not do architecture planning yet.",
            mode="auto",
        )

        assert result.chosen_path == "Prior Knowledge only"
        assert result.prior_knowledge is not None
        assert result.architecture is None

    def test_run_architect_auto_routes_existing_handoff_directly_to_architecture(self) -> None:
        prior = run_prior_knowledge(
            "Battery degradation forecasting depends on temperature and cycle count, with weekly operational batching."
        )

        result = run_architect(prior.to_handoff_text(), mode="auto")

        assert result.chosen_path == "Architecture Focus only"
        assert result.prior_knowledge is None
        assert result.architecture is not None

    def test_run_architecture_focus_rejects_malformed_prior_handoff(self) -> None:
        malformed_handoff = "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF\nSystem Summary:\n- Battery degradation forecasting"

        with pytest.raises(ValueError, match="Malformed GPArchitect prior-knowledge handoff block"):
            run_architecture_focus(malformed_handoff)
