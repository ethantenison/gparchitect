# GPArchitect Testing Strategy

This document defines the sustainable testing regime for GPArchitect.

It follows the repository testing rules in `.github/copilot-instructions.md`:

- test behavior and invariants rather than implementation detail
- prefer the lowest effective level
- avoid duplicating the same coverage across unit, integration, and end-to-end tests
- use canonical patterns instead of permutation grids

---

## Layered Test Plan

Use four layers, each with a narrow job.

### 1. Translator Unit Tests

Purpose:
- verify natural language becomes the correct `GPSpec`

What belongs here:
- model class selection
- shared mean parsing
- per-output and per-task mean parsing
- feature-group to column mapping
- explicit kernel attributes in the DSL
- omitted optional attributes staying unset in the DSL

What does not belong here:
- fitting success
- BoTorch object type assertions

### 2. Validator Unit Tests

Purpose:
- verify valid and invalid DSL combinations are accepted or rejected correctly

What belongs here:
- invalid `output_means` targets
- multitask-specific constraints
- incompatible feature-group definitions

### 3. Builder Integration Tests

Purpose:
- verify a validated DSL builds the expected BoTorch and GPyTorch model structure

What belongs here:
- `SingleTaskGP`, `MultiTaskGP`, and `ModelListGP` construction
- mean-module wiring
- likelihood selection
- covariance structure and kernel parameter propagation

What does not belong here:
- natural-language parsing behavior already covered in translator tests
- large end-to-end datasets

### 4. Public API End-to-End Tests

Purpose:
- verify representative user instructions survive the full `run_gparchitect` pipeline

What belongs here:
- one canonical successful case per supported architecture
- one canonical explicit-parameter case
- one canonical default-fallback case for omitted optional parameters
- a small number of realistic multi-feature-group instructions

What does not belong here:
- exhaustive kernel by mean by model-class combinations
- repeated assertions that already exist in unit or builder tests

---

## Assertion Rules

When testing instruction fidelity, assert the contract at the correct layer.

For model class:
- translator: `spec.model_class`
- builder or end-to-end: built model type

For feature groups:
- translator or end-to-end log snapshot: `feature_indices` and `kernel_type`

For means:
- translator: `spec.mean` or `spec.output_means`
- builder or end-to-end: mean-module behavior and representative module type

For explicit kernel attributes:
- translator or end-to-end log snapshot: `rq_alpha`, `period_length`, `polynomial_power`, and similar fields
- builder: one representative object-level assertion when wiring matters

For omitted optional attributes:
- assert the DSL snapshot leaves the field as `None`
- avoid asserting version-sensitive raw library default values unless GPArchitect sets that default itself

---

## Growth Rules

When adding a new user-facing modeling capability:

1. Add one translator test for parsing the new instruction pattern.
2. Add one validator test only if the new capability introduces invalid combinations.
3. Add one builder integration test if the DSL changes model wiring.
4. Add one end-to-end test only if the feature defines a canonical successful user path.

Do not add a cross-product matrix over:

- kernel families
- priors
- ARD settings
- mean functions
- multitask settings

Add a new end-to-end case only when it protects a distinct public contract.

---

## Current Canonical Coverage

The intended steady-state public API coverage is:

- `SingleTaskGP` with explicit mean and explicit kernel parameter
- `ModelListGP` with per-output means
- `MultiTaskGP` with per-task means
- one omitted-parameter case showing default fallback through an unset DSL field
- a small number of realistic multi-feature-group examples