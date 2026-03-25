"""Tests for GPArchitect custom agent orchestration files.

These tests validate the planning-agent contract stored in workspace agent
customization files. They verify that the Planning Orchestrator delegates to
the expected specialists, preserves the two-stage routing behavior, and uses a
consistent prior-knowledge handoff between stages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
REQUIRED_FRONTMATTER_KEYS = {
    "name": str,
    "description": str,
    "tools": list,
    "agents": list,
    "argument-hint": str,
    "user-invocable": bool,
    "disable-model-invocation": bool,
}


def _read_agent(agent_filename: str) -> tuple[dict[str, object], str]:
    """Return parsed frontmatter and body for an agent customization file."""

    content = (AGENTS_DIR / agent_filename).read_text(encoding="utf-8")
    _, frontmatter_block, body = content.split("---\n", maxsplit=2)
    return _parse_frontmatter(frontmatter_block), body


def _iter_agent_files() -> list[Path]:
    """Return all workspace custom agent files in a deterministic order."""

    return sorted(AGENTS_DIR.glob("*.agent.md"))


def _parse_frontmatter(frontmatter_block: str) -> dict[str, object]:
    """Parse the limited YAML frontmatter shape used by workspace agent files."""

    frontmatter: dict[str, object] = {}

    for raw_line in frontmatter_block.splitlines():
        if not raw_line.strip():
            continue

        key, raw_value = raw_line.split(":", maxsplit=1)
        value = raw_value.strip()

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            items = [] if not inner else [item.strip().strip('"') for item in inner.split(",")]
            frontmatter[key] = items
            continue

        if value in {"true", "false"}:
            frontmatter[key] = value == "true"
            continue

        if value.startswith('"') and value.endswith('"'):
            frontmatter[key] = value[1:-1]
            continue

        frontmatter[key] = value

    return frontmatter


def _get_string(frontmatter: dict[str, object], key: str) -> str:
    """Return a string frontmatter field after validating its type."""

    value = frontmatter[key]
    assert isinstance(value, str), f"Frontmatter field '{key}' must be a string"
    return value


def _get_string_list(frontmatter: dict[str, object], key: str) -> list[str]:
    """Return a list[str] frontmatter field after validating its type."""

    value = frontmatter[key]
    assert isinstance(value, list), f"Frontmatter field '{key}' must be a list"
    assert all(isinstance(item, str) for item in value), f"Frontmatter field '{key}' must contain only strings"
    return value


class TestPlanningOrchestratorAgent:
    """Validate the two-stage planning orchestrator contract."""

    def test_orchestrator_frontmatter_declares_both_specialist_agents(self) -> None:
        frontmatter, _ = _read_agent("planning-orchestrator.agent.md")

        assert _get_string(frontmatter, "name") == "Planning Orchestrator"
        assert _get_string_list(frontmatter, "tools") == ["read", "agent", "execute"]
        assert _get_string_list(frontmatter, "agents") == ["Prior Knowledge", "Architecture Focus"]
        assert frontmatter["user-invocable"] is True

    def test_orchestrator_defaults_to_two_stage_flow_for_raw_planning_input(self) -> None:
        _, body = _read_agent("planning-orchestrator.agent.md")

        assert "1. Prior Knowledge only" in body
        assert "2. Architecture Focus only" in body
        assert "3. Prior Knowledge followed by Architecture Focus" in body
        assert "## Route Through Both Stages" in body
        assert (
            "Run Prior Knowledge followed by Architecture Focus when the user wants end-to-end planning "
            "from raw prior knowledge to architecture-planning output." in body
        )
        assert "This is the default path when:" in body
        assert "- the input is raw or partially structured prior knowledge, and" in body
        assert "- the user is asking for planning rather than only elicitation." in body

    def test_two_stage_handoff_contract_is_shared_across_specialists(self) -> None:
        orchestrator_frontmatter, orchestrator_body = _read_agent("planning-orchestrator.agent.md")
        prior_frontmatter, prior_body = _read_agent("prior_knowledge.agent.md")
        architecture_frontmatter, architecture_body = _read_agent("architecture-focus.agent.md")

        assert (AGENTS_DIR / "planning-orchestrator.agent.md").is_file()
        assert (AGENTS_DIR / "prior_knowledge.agent.md").is_file()
        assert (AGENTS_DIR / "architecture-focus.agent.md").is_file()

        assert _get_string_list(orchestrator_frontmatter, "agents") == ["Prior Knowledge", "Architecture Focus"]
        assert _get_string_list(prior_frontmatter, "agents") == ["Architecture Focus"]
        assert _get_string_list(architecture_frontmatter, "agents") == []

        assert "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in prior_body
        assert "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in prior_body
        assert "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in architecture_body
        assert "END GPARCHITECT PRIOR KNOWLEDGE HANDOFF" in architecture_body
        assert "If Prior Knowledge produces a GPArchitect prior-knowledge handoff" in orchestrator_body
        assert "pass that handoff into Architecture Focus" in orchestrator_body
        assert "When both stages run, include:" in orchestrator_body
        assert "- the GPArchitect prior-knowledge handoff" in orchestrator_body
        assert "- the GPArchitect architecture handoff" in orchestrator_body

    def test_orchestrator_documents_cli_runtime_bridge(self) -> None:
        frontmatter, body = _read_agent("planning-orchestrator.agent.md")

        assert "execute" in _get_string_list(frontmatter, "tools")
        assert "## Runtime Bridge" in body
        assert "gparchitect-plan auto" in body
        assert "authoritative structured artifact" in body


class TestAgentFrontmatterSchema:
    """Validate shared frontmatter schema across all workspace agents."""

    @pytest.mark.parametrize("agent_path", _iter_agent_files(), ids=lambda path: path.name)
    def test_all_agent_files_use_required_frontmatter_schema(self, agent_path: Path) -> None:
        frontmatter, body = _read_agent(agent_path.name)

        assert body.strip(), f"{agent_path.name} must include a non-empty body"

        for key, expected_type in REQUIRED_FRONTMATTER_KEYS.items():
            assert key in frontmatter, f"{agent_path.name} is missing required frontmatter key: {key}"
            assert isinstance(frontmatter[key], expected_type), (
                f"{agent_path.name} frontmatter key '{key}' must be of type {expected_type.__name__}"
            )

        assert _get_string(frontmatter, "name").strip(), f"{agent_path.name} must define a non-empty name"
        assert _get_string(frontmatter, "description").strip(), (
            f"{agent_path.name} must define a non-empty description"
        )
        assert _get_string(frontmatter, "description").startswith("Use "), (
            f"{agent_path.name} description should start with 'Use ' for discoverability"
        )
        assert _get_string(frontmatter, "argument-hint").strip(), (
            f"{agent_path.name} must define a non-empty argument-hint"
        )

        for tool_name in _get_string_list(frontmatter, "tools"):
            assert isinstance(tool_name, str) and tool_name.strip(), (
                f"{agent_path.name} contains an invalid tool entry: {tool_name!r}"
            )

        for agent_name in _get_string_list(frontmatter, "agents"):
            assert isinstance(agent_name, str) and agent_name.strip(), (
                f"{agent_path.name} contains an invalid delegated agent entry: {agent_name!r}"
            )

    @pytest.mark.parametrize("agent_path", _iter_agent_files(), ids=lambda path: path.name)
    def test_all_delegated_agents_resolve_to_workspace_agent_names(self, agent_path: Path) -> None:
        available_agent_names = {
            _get_string(_read_agent(candidate.name)[0], "name")
            for candidate in _iter_agent_files()
        }
        frontmatter, _ = _read_agent(agent_path.name)

        for delegated_agent_name in _get_string_list(frontmatter, "agents"):
            assert delegated_agent_name in available_agent_names, (
                f"{agent_path.name} delegates to unknown agent: {delegated_agent_name}"
            )
