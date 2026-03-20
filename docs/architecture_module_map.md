# GPArchitect Architecture Module Map

This document describes the architecture boundaries for GPArchitect.
Refer to `.github/copilot-instructions.md` for normative rules, and
`docs/quickstart.md` for user-facing documentation.

---

## Pipeline Overview

```
Natural language
      │
      ▼
┌─────────────────────┐
│   Translator        │  src/gparchitect/translator/
│   NL → GPSpec DSL   │
└─────────┬───────────┘
          │ GPSpec
          ▼
┌─────────────────────┐
│   Validation        │  src/gparchitect/validation/
│   GPSpec → Result   │
└─────────┬───────────┘
          │ validated GPSpec
          ▼
┌─────────────────────┐
│   Model Builder     │  src/gparchitect/builders/
│   GPSpec → BoTorch  │
└─────────┬───────────┘
          │ BoTorch model
          ▼
┌─────────────────────┐
│   Fitting           │  src/gparchitect/fitting/
│   Model → FitResult │
└─────────┬───────────┘
          │ FitResult (success/failure)
          ▼
  [success] ─────────────────────────► Return model
  [failure]
          │
          ▼
┌─────────────────────┐
│   Revision          │  src/gparchitect/revision/
│   GPSpec → GPSpec   │
└─────────┬───────────┘
          │ revised GPSpec
          └──► Validation (retry loop)

All stages write to:
┌─────────────────────┐
│   Logging           │  src/gparchitect/logging/
│   ExperimentLog     │
└─────────────────────┘
```

---

## Module Responsibilities

### `src/gparchitect/dsl/`

**Responsibility**: Define the GP DSL schema as Pydantic models.

**Key types**:
- `GPSpec` — root DSL object
- `FeatureGroupSpec` — a group of input features with a shared kernel
- `KernelSpec` — a kernel or composition of kernels
- `NoiseSpec` — noise model specification
- `PriorSpec` — hyperparameter prior specification

**Boundary rule**: This module has NO dependencies on other gparchitect modules.
It must remain import-free from the rest of the codebase.

---

### `src/gparchitect/translator/`

**Responsibility**: Convert natural-language strings to `GPSpec` objects.

**Key functions**:
- `translate_to_dsl(instruction, input_dim, output_dim, task_feature_index) → GPSpec`

**Boundary rule**: The translator must NEVER import from `builders`, `fitting`,
`revision`, or `logging`. It may only import from `dsl`.

---

### `src/gparchitect/validation/`

**Responsibility**: Validate a `GPSpec` for structural and semantic correctness.

**Key types**:
- `ValidationResult` — errors and warnings lists

**Key functions**:
- `validate_dsl(spec) → ValidationResult`
- `validate_or_raise(spec) → None`

**Boundary rule**: The validator must NEVER import from `builders`, `fitting`,
`revision`, or `logging`. It may only import from `dsl`.

---

### `src/gparchitect/builders/`

**Responsibility**: Construct BoTorch GP models from a validated `GPSpec`.

**Key functions**:
- `build_model_from_dsl(spec, train_X, train_Y) → BoTorch model`
- `prepare_data(dataframe, input_columns, output_columns, task_column) → DataBundle`

**Boundary rule**: Builders import `dsl` and `torch`/`botorch`/`gpytorch`.
They must NOT import from `translator`, `fitting`, `revision`, or `logging`.

---

### `src/gparchitect/fitting/`

**Responsibility**: Fit a BoTorch model and run a basic prediction check.

**Key types**:
- `FitResult` — success flag, model, MLL value, error message

**Key functions**:
- `fit_and_validate(model, train_X, train_Y) → FitResult`

**Boundary rule**: Fitting imports `torch`/`botorch`/`gpytorch` only.
It must NOT import from `translator`, `builders`, `revision`, or `logging`.

---

### `src/gparchitect/revision/`

**Responsibility**: Revise a `GPSpec` to recover from fitting/construction failures.

**Key types**:
- `RevisionResult` — revised spec, rationale, strategy name

**Key functions**:
- `revise_dsl(spec, error_message, attempt) → RevisionResult | None`

**Boundary rule**: Revision imports `dsl` only. It must NOT import from `translator`,
`builders`, `fitting`, or `logging`.

---

### `src/gparchitect/logging/`

**Responsibility**: Record all pipeline events as a JSON-serializable `ExperimentLog`.

**Key types**:
- `ExperimentLog` — full history of a pipeline run
- `AttemptRecord` — one attempt within a run

**Key functions**:
- `summarize_attempts(log) → str`

**Boundary rule**: Logging imports `dsl` only (for `GPSpec.model_dump`).
It must NOT import from `translator`, `builders`, `fitting`, or `revision`.

---

### `src/gparchitect/api.py`

**Responsibility**: Top-level orchestration API, combining all pipeline stages.

This is the only module permitted to import from ALL other subpackages.

---

### `src/gparchitect/cli.py`

**Responsibility**: Command-line interface wrapping `api.run_gparchitect`.

This module imports from `api` only and handles CSV I/O and terminal output.

---

## Dependency Matrix

| Module       | dsl | translator | validation | builders | fitting | revision | logging |
|:------------ |:---:|:----------:|:----------:|:--------:|:-------:|:--------:|:-------:|
| translator   | ✓   |            |            |          |         |          |         |
| validation   | ✓   |            |            |          |         |          |         |
| builders     | ✓   |            |            |          |         |          |         |
| fitting      |     |            |            |          |         |          |         |
| revision     | ✓   |            |            |          |         |          |         |
| logging      | ✓   |            |            |          |         |          |         |
| api          | ✓   | ✓          | ✓          | ✓        | ✓       | ✓        | ✓       |
| cli          |     |            | ✓          | ✓        |         |          |         |

`api` is the **only** module that may import from all others.
