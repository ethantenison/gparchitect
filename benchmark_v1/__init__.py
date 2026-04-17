"""
benchmark_v1 — GPArchitect reproducibility and comparison suite.

Purpose:
    Provides a deterministic benchmark suite to evaluate whether GPArchitect's
    natural-language-to-DSL model construction improves predictive performance
    and/or robustness relative to strong GP baselines.

Structure:
    registry.py             — benchmark registry: datasets, prompts, seeds, noise levels
    datasets/               — dataset generators (Tier 1: BoTorch functions, Tier 2: synthetic tabular)
    prompts/                — prompt variant definitions per dataset
    baselines/              — baseline model configurations
    run_benchmark.py        — reproducible runner CLI
    analyze_results.py      — aggregation and report generation

Usage::

    python -m benchmark_v1.run_benchmark --output-dir results/benchmark_v1
    python -m benchmark_v1.analyze_results results/benchmark_v1 --report docs/benchmark_v1_report.md
"""

from __future__ import annotations
