# GitHub Copilot Instructions — GPArchitect

These instructions are strict. Generated code and documentation must follow them.

GPArchitect is a Python library for building Gaussian Process models from
natural-language instructions via a formal intermediate DSL using BoTorch
and GPyTorch.

---

## Core Architectural Rule

Natural language must NEVER directly construct models.

All model construction must follow this pipeline:

Natural language → GP DSL → Validation → Model Builder → Fit → Validation → Recovery

The DSL representation is the single source of truth.

---

## Documentation Source of Truth

Keep this file for normative rules only.

- Use this file for must and must-not guidance.
- Use `docs/architecture_module_map.md` for architecture boundaries.
- Use `README.md` and `docs/quickstart.md` for user-facing documentation.

When functionality changes:

- Update this file only if rules change
- Update architecture docs if boundaries change
- Update README/quickstart if usage changes

---

## Architecture and Design Rules

GPArchitect is a compiler-style system with strict separation of concerns:

1. Natural language translation
2. DSL schema and representation
3. DSL validation
4. Model construction from DSL
5. Fitting and prediction
6. DSL revision and recovery
7. Logging and experiment history
8. User-facing API and CLI

### Key Principles

- DSL is the authoritative specification
- No hidden state or implicit behavior
- Deterministic model construction
- Explicit data flow between modules
- Reproducibility of results
- Inspectability of decisions

---

## DSL Requirements

The GP DSL must be:

- Structured and typed (dataclasses or Pydantic)
- Fully parseable
- Validatable before execution
- JSON-serializable
- Human-readable
- Stable across versions
- Independent of natural-language phrasing

The DSL must represent:

- Model class
- Input feature groups
- Kernel types per group
- Kernel compositions (additive, multiplicative)
- ARD usage
- Priors when specified
- Noise assumptions
- Multitask structure
- Optional constraints

---

## Language and Style

- Target Python 3.13+
- Use native typing only (`list[str]`, `dict[str, float]`, `X | None`)
- Use `from __future__ import annotations`
- Prefer clarity over cleverness
- Avoid one-letter variable names
- Use logging instead of print

---

## Documentation Requirements

Every module must include a top-level docstring explaining:

- Purpose
- Role in the GPArchitect pipeline
- Inputs and outputs
- Non-obvious design decisions
- What the module does not do

All public APIs require Google-style docstrings.

---

## Code Quality

Avoid:

- Unused imports or variables
- Bare except blocks
- Silent failure paths
- Commented-out code
- Global mutable state

Max line length: 120 characters.

---

## Testing Expectations

Add tests for core pipeline invariants:

- Natural language → DSL translation
- DSL validation (valid and invalid cases)
- Model construction from DSL
- Failure recovery via DSL revision
- Fallback-to-default behavior

Use small, deterministic synthetic datasets only.

---

### Test Design Principles

Write pytest tests that are minimal, high-value, and easy to maintain.

- Test **behavior and invariants**, not implementation details
- Prefer the **lowest effective level**:
  - Unit tests for DSL logic, validation, and transformations
  - Integration tests only for model construction and fitting boundaries
- Do not duplicate coverage across unit, integration, and end-to-end tests
- One test = **one behavior or one failure mode**
- Before adding a test, ask: *what unique regression does this catch?*

---

### Controlling Test Explosion

- Prefer **representative cases** over exhaustive combinations
- Do not generate combinatorial test grids over:
  - kernels
  - priors
  - ARD configurations
  - multitask settings
- Use parametrization only for **small, meaningful matrices**
- Avoid testing all DSL permutations; test **canonical patterns and edge cases**

---

### Fixtures and Test Data

- Use **small, explicit fixtures** only for shared setup
- Prefer **factories/builders** for DSL objects and datasets
- Avoid large or implicit fixture chains (“fixture soup”)
- Keep all test data:
  - minimal
  - readable
  - deterministic

---

### Mocks, Fakes, and Boundaries

- Prefer **fakes** for repositories and internal services
- Use mocks only to verify **external interaction contracts**
- Do not mock core DSL or model-building logic

---

### Unit Test Constraints

Unit tests must:

- Run fast (no heavy GP fitting loops unless trivial)
- Be deterministic (no randomness without fixed seeds)
- Avoid:
  - real training loops when unnecessary
  - large datasets
  - external I/O

---

### Integration Test Scope

Integration tests should only verify:

- DSL → model construction correctness
- Compatibility with BoTorch/GPyTorch APIs
- Successful fitting on small datasets

Keep integration tests **few and targeted**.

---

### Regression Discipline

- Every real bug must include a **regression test**
- Do not add redundant tests for already-covered behavior
- Prefer adding tests at the **lowest level that would have caught the bug**

---

### Anti-Patterns (Must Avoid)

- Testing internal/private methods directly
- Snapshotting large DSL or model objects
- Asserting on call order or internal helper usage
- Duplicating the same test across multiple layers
- Large parameter sweeps disguised as tests

---

## BoTorch and GPyTorch Rules

- Prefer SingleTaskGP, MultiTaskGP, ModelListGP
- Avoid custom ExactGP subclasses unless necessary
- Use BoTorch defaults when parameters are unspecified
- Apply priors only when explicitly requested
- Ensure compatibility with BoTorch fitting utilities

---

## Failure Recovery

If construction, fitting, or prediction fails:

1. Capture error
2. Revise DSL specification (not natural language)
3. Revalidate DSL
4. Retry model construction
5. Log all revisions and rationale

Allowed recovery actions include:

- Simplifying kernels
- Removing invalid priors
- Disabling ARD
- Switching model type
- Using default noise assumptions

All revisions must be recorded.

---

## Non-Goals

- No web application or UI
- No categorical or binary input support in v1
- Focus on continuous inputs and optional task variables
