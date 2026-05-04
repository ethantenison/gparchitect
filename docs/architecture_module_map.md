# GPArchitect Architecture Module Map

This document describes the architecture boundaries for GPArchitect.
Refer to `.github/copilot-instructions.md` for normative rules, and
`docs/quickstart.md` for user-facing documentation.

---

## Pipeline Overview

```
Natural language planning input
  │
  ▼
┌─────────────────────┐
│     Planning        │  src/gparchitect/planning/
│ prior knowledge and │
│ architecture intent │
└─────────┬───────────┘
      │ planning handoffs
      ▼
Natural language model instruction
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

### Grouping Policy

GPArchitect treats feature grouping as part of the DSL translation contract.

- Each natural-language kernel mention maps to one `FeatureGroupSpec`.
- Column references are resolved only against the provided `input_columns`.
- A feature stays in the group named by its local kernel clause; it must not bleed into
  the next kernel mention.
- Time-like features such as `time`, `date`, `month`, `month_index`, or `time_index`
  remain singleton groups when they are described in their own kernel clause.
- Multiple features may share a group when they are named inside the same kernel clause.
- When no feature groups are resolved from the instruction, translation falls back to a
  single all-input feature group.
- When multiple feature groups are present and the instruction does not explicitly request
  additive or multiplicative composition, the default inter-group composition is
  hierarchical: main effects plus pairwise interactions.
- ARD defaults to `True` for kernel families that support per-dimension lengthscales and
  switches to `False` only when the instruction explicitly disables it.

Kernel construction follows these builder-side rules:

- Leaf kernels are wrapped in `ScaleKernel`.
- Additive compositions keep per-term `ScaleKernel` wrappers and do not add an extra
  outer scale around the sum.
- Multiplicative compositions use a single outer `ScaleKernel` around the full product
  and do not scale each factor independently.
- Hierarchical interaction terms are built as scaled products, while their main effects
  remain individually scaled additive terms.

### `src/gparchitect/dsl/`

**Responsibility**: Define the GP DSL schema as Pydantic models.

**Key types**:
- `GPSpec` — root DSL object
- `FeatureGroupSpec` — a group of input features with a shared kernel
- `KernelSpec` — a kernel or composition of kernels, with optional `time_varying` spec
- `MeanSpec` — a shared or per-output mean-function specification
- `NoiseSpec` — noise model specification
- `PriorSpec` — hyperparameter prior specification
- `ExecutionSpec` — explicit preprocessing and outcome-transform semantics, including
  `recency_filtering` (Tier 1) and `input_warping` (Tier 2)
- `RecencyFilteringSpec` — Tier 1 dataset-truncation spec (formerly mislabeled "recency weighting")
- `TimeVaryingSpec` — Tier 2 time-varying hyperparameter spec for `KernelSpec.time_varying`
- `InputWarpingSpec` — Tier 2 Kumaraswamy input-warp spec for `ExecutionSpec.input_warping`

For the current validated contract:

- `MultiTaskGP` uses long-format data with one observed output column and a task indicator.
- `MultiTaskGP` requires an explicit task domain in `GPSpec.task_values`.
- Supported priors are a validated subset (`Normal`, `LogNormal`, `Gamma`, `HalfCauchy`,
  and `Uniform`) that the builder applies directly; unsupported
  prior placements must be rejected during validation.
- Execution semantics such as input scaling and outcome standardization live in
  `GPSpec.execution` so they are machine-readable contract surface.
- `recency_filtering` is dataset truncation (Tier 1), not true likelihood weighting.
  The canonical field name is `recency_filtering`; the old name `recency_weighting` is removed.
- `time_varying` on a `KernelSpec` (Tier 2) wraps the base kernel with a linear modulation.
  Only `parameterization="linear"` is supported.  Leaf and composed kernels (with children)
  may carry `time_varying`.
- `input_warping` on `ExecutionSpec` (Tier 2) applies a Kumaraswamy CDF warp via BoTorch's
  `Warp` input transform.  Input scaling should be active for the warp to operate in `[0, 1]`.
- `heteroskedastic_noise=True` is currently rejected by the validator as a forward-compatibility
  placeholder.

**Boundary rule**: This module has NO dependencies on other gparchitect modules.
It must remain import-free from the rest of the codebase.

---

### `src/gparchitect/planning/`

**Responsibility**: Produce structured planning artifacts that capture prior knowledge,
architecture implications, and orchestration routing before GP DSL construction.

**Key types**:
- `PriorKnowledgeHandoff` — structured prior-knowledge artifact
- `ArchitectureHandoff` — structured architecture-planning artifact
- `PlanningRunResult` — route selection and stage outputs

**Key functions**:
- `run_prior_knowledge(input_text) -> PriorKnowledgeHandoff`
- `run_architecture_focus(handoff) -> ArchitectureHandoff`
- `run_architect(input_text, mode) -> PlanningRunResult`

**Boundary rule**: Planning must NOT import from `translator`, `validation`, `builders`,
`fitting`, `revision`, or `logging`. It sits upstream of DSL construction and must stop
at planning artifacts rather than producing a model or GP DSL directly.

---

### `src/gparchitect/translator/`

**Responsibility**: Convert natural-language strings to `GPSpec` objects.

**Key functions**:
- `translate_to_dsl(instruction, input_dim, output_dim, task_feature_index) → GPSpec`

Translation rules that matter for contract alignment:

- `ModelListGP` without output-specific grouping falls back to one shared all-input
  feature-group specification reused across output models.
- When targeted multitask means are parsed from natural language, translation records
  the explicit task values in `GPSpec.task_values`.
- Translation can parse supported prior phrases into `PriorSpec` objects attached to
  kernel or noise targets.

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
- `apply_recency_filtering(train_X, train_Y, spec) → (train_X, train_Y)` — Tier 1 dataset truncation

Builder responsibilities include:

- Converting `MeanSpec` into GPyTorch mean modules
- Applying optional per-output means for `ModelListGP`
- Applying optional per-task means for `MultiTaskGP`
- Applying the supported validated subset of DSL priors to kernels and learnable noise
- Respecting the execution semantics in `GPSpec.execution`
- Selecting the likelihood implementation from the DSL noise spec and model class
- **Tier 2**: Wrapping base kernels in `TimeVaryingKernel` when `KernelSpec.time_varying` is set.
  The wrapper adds learned bias and slope parameters for smooth outputscale or lengthscale
  modulation as a function of the designated time feature.
- **Tier 2**: Attaching a BoTorch `Warp` input transform when `ExecutionSpec.input_warping` is set.
  The warp is applied to the designated time feature dimension only.

**Key modules**:
- `builder.py` — main model construction and composition
- `recency.py` — Tier 1 recency filtering (dataset truncation before fitting)
- `time_varying_kernel.py` — Tier 2 time-varying kernel wrapper modules
- `changepoint_kernel.py` — Tier 1 changepoint kernel implementation
- `data.py` — data preparation utilities

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

**Responsibility**: Command-line interface wrapping `api.run_gparchitect` and the
executable planning runtime.

This module imports from `api` for model construction and `planning` for pre-DSL
planning commands. It handles CSV I/O, planning text input, and terminal output.

---

## Dependency Matrix

| Module       | dsl | planning | translator | validation | builders | fitting | revision | logging |
|:------------ |:---:|:--------:|:----------:|:----------:|:--------:|:-------:|:--------:|:-------:|
| planning     |     |          |            |            |          |         |          |         |
| translator   | ✓   |          |            |            |          |         |          |         |
| validation   | ✓   |          |            |            |          |         |          |         |
| builders     | ✓   |          |            |            |          |         |          |         |
| fitting      |     |          |            |            |          |         |          |         |
| revision     | ✓   |          |            |            |          |         |          |         |
| logging      | ✓   |          |            |            |          |         |          |         |
| api          | ✓   |          | ✓          | ✓          | ✓        | ✓       | ✓        | ✓       |
| cli          |     | ✓        | ✓          | ✓          | ✓        |         |          |         |

`api` is the **only** module that may import from all others.
