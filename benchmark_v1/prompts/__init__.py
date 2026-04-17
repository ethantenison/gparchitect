"""
benchmark_v1.prompts — package for benchmark prompt variant definitions.

Each dataset in the registry has three prompt variants:
    aligned     — instruction matches the dataset's known generating structure
    vague       — generic/default instruction with no structural hints
    misleading  — instruction contains a plausible but incorrect structural hint
"""

from __future__ import annotations
