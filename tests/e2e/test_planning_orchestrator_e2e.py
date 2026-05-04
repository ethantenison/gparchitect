"""End-to-end contract tests for Planning Orchestrator prompt routing.

These tests exercise realistic prompt fixtures against the documented routing
policy in the workspace custom agent files. They do not execute the VS Code
agent runtime; instead, they verify that representative user inputs map to the
expected specialist sequence and handoff contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "planning_orchestrator_prompts.json"


def _read_agent_body(agent_filename: str) -> str:
    """Return only the markdown body for a workspace agent file."""

    content = (AGENTS_DIR / agent_filename).read_text(encoding="utf-8")
    _, _, body = content.split("---\n", maxsplit=2)
    return body


@pytest.fixture(scope="module")
def orchestrator_prompt_fixtures() -> list[dict[str, object]]:
    """Load realistic prompt fixtures that describe routing scenarios."""

    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def orchestrator_body() -> str:
    """Load the Planning Orchestrator agent body."""

    return _read_agent_body("planning-orchestrator.agent.md")


@pytest.fixture(scope="module")
def prior_knowledge_body() -> str:
    """Load the Prior Knowledge agent body."""

    return _read_agent_body("prior_knowledge.agent.md")


@pytest.fixture(scope="module")
def architecture_focus_body() -> str:
    """Load the Architecture Focus agent body."""

    return _read_agent_body("architecture-focus.agent.md")


def _select_path(prompt_fixture: dict[str, object]) -> str:
    """Apply the orchestrator's documented routing contract to a prompt fixture."""

    if prompt_fixture["contains_handoff"]:
        return "Architecture Focus only"

    if prompt_fixture["planning_requested"]:
        return "Prior Knowledge -> Architecture Focus"

    return "Prior Knowledge only"


def _expected_stage_outputs(selected_path: str) -> list[str]:
    """Return the expected deliverables for a selected orchestration path."""

    if selected_path == "Prior Knowledge -> Architecture Focus":
        return ["GPArchitect prior-knowledge handoff", "GPArchitect architecture handoff"]

    if selected_path == "Architecture Focus only":
        return ["GPArchitect architecture handoff"]

    return ["GPArchitect prior-knowledge handoff"]


@pytest.mark.parametrize("fixture_index", [0, 1, 2])
def test_prompt_fixtures_route_to_expected_stage_sequence(
    fixture_index: int,
    orchestrator_prompt_fixtures: list[dict[str, object]],
    orchestrator_body: str,
) -> None:
    """Representative prompts should map to the documented route selection."""

    prompt_fixture = orchestrator_prompt_fixtures[fixture_index]

    assert "Choose the path based on the user's input quality and intent." in orchestrator_body
    assert "### Route To Prior Knowledge First" in orchestrator_body
    assert "### Route Directly To Architecture Focus" in orchestrator_body
    assert "### Route Through Both Stages" in orchestrator_body

    selected_path = _select_path(prompt_fixture)
    assert selected_path == prompt_fixture["expected_path"]


@pytest.mark.parametrize("fixture_index", [0, 1, 2])
def test_prompt_fixtures_preserve_expected_delegation_contract(
    fixture_index: int,
    orchestrator_prompt_fixtures: list[dict[str, object]],
    orchestrator_body: str,
    prior_knowledge_body: str,
    architecture_focus_body: str,
) -> None:
    """Representative prompts should imply the expected delegated stages and outputs."""

    prompt_fixture = orchestrator_prompt_fixtures[fixture_index]
    selected_path = _select_path(prompt_fixture)

    if "Prior Knowledge" in prompt_fixture["expected_agents"]:
        assert "If Prior Knowledge is needed, invoke the Prior Knowledge agent first." in orchestrator_body
        assert "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in prior_knowledge_body
        assert "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in prior_knowledge_body

    if "Architecture Focus" in prompt_fixture["expected_agents"]:
        assert (
            "If the user already provides a valid prior-knowledge handoff, skip Prior Knowledge and invoke Architecture Focus directly."
            in orchestrator_body
        )
        assert "BEGIN GPARCHITECT ARCHITECTURE HANDOFF" in architecture_focus_body
        assert "END GPARCHITECT ARCHITECTURE HANDOFF" in architecture_focus_body

    expected_outputs = _expected_stage_outputs(selected_path)
    for expected_output in expected_outputs:
        assert expected_output in orchestrator_body or expected_output in architecture_focus_body


def test_prompt_fixture_file_covers_all_supported_orchestrator_paths(
    orchestrator_prompt_fixtures: list[dict[str, object]],
) -> None:
    """The fixture set should cover each orchestrator path exactly once."""

    observed_paths = {fixture["expected_path"] for fixture in orchestrator_prompt_fixtures}

    assert observed_paths == {
        "Prior Knowledge only",
        "Architecture Focus only",
        "Prior Knowledge -> Architecture Focus",
    }
