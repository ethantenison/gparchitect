"""
benchmark_v1.prompts.variants — prompt variant definitions for each benchmark dataset.

Purpose:
    Defines three natural-language prompt variants per benchmark dataset:
        - aligned: instruction matches the known generating structure
        - vague: generic/default instruction with no structural hints
        - misleading: plausible but structurally incorrect instruction

Role in benchmark pipeline:
    Prompts → Registry → Runner → GPArchitect translator → DSL → model

Non-obvious design decisions:
    - Prompts are stored as plain strings, not templates; they are fixed at
      definition time so that benchmarks are fully reproducible.
    - The ``aligned`` variant is the one most likely to benefit from
      GPArchitect's instruction pathway.
    - The ``vague`` variant is the control; it should produce roughly the same
      result as a default SingleTaskGP baseline.
    - The ``misleading`` variant tests robustness of the revision/recovery
      mechanism.

What this module does NOT do:
    - It does not call any LLM or external service.
    - It does not validate prompts against DSL schema.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVariants:
    """A set of three prompt variants for a single benchmark dataset.

    Attributes:
        dataset_name: Identifier matching the registry entry.
        aligned: Instruction that matches the dataset's known structure.
        vague: Generic instruction with no structural guidance.
        misleading: Instruction with a plausible but incorrect structural hint.
    """

    dataset_name: str
    aligned: str
    vague: str
    misleading: str

    def as_dict(self) -> dict[str, str]:
        """Return the three variants as a name → prompt mapping.

        Returns:
            Dict with keys ``"aligned"``, ``"vague"``, ``"misleading"``.
        """
        return {
            "aligned": self.aligned,
            "vague": self.vague,
            "misleading": self.misleading,
        }


# ---------------------------------------------------------------------------
# Tier 2 — named-column synthetic datasets
# ---------------------------------------------------------------------------

ADDITIVE_PROMPTS = PromptVariants(
    dataset_name="additive",
    aligned=(
        "Use an additive Matern52 kernel with ARD over all continuous inputs. "
        "The signal is a sum of smooth terms; irrelevant features should be "
        "down-weighted automatically."
    ),
    vague=(
        "Fit a GP to this tabular data."
    ),
    misleading=(
        "Use a periodic kernel because the data has seasonal patterns. "
        "Apply it to all inputs."
    ),
)

PERIODIC_DECAY_PROMPTS = PromptVariants(
    dataset_name="periodic_decay",
    aligned=(
        "Use a multiplicative combination of a Periodic kernel on seasonality_index "
        "and an ExponentialDecay kernel on system_age. "
        "The signal has both a seasonal oscillation and a long-term decay."
    ),
    vague=(
        "Fit a standard GP to this time-series-like regression data."
    ),
    misleading=(
        "Use an additive RBF kernel on all inputs. "
        "The features are independently contributing."
    ),
)

INTERACTION_PROMPTS = PromptVariants(
    dataset_name="interaction",
    aligned=(
        "Use a multiplicative RBF kernel on material_hardness and process_temperature "
        "to capture their interaction, and ignore cooldown_rate as it is not predictive."
    ),
    vague=(
        "Fit a GP regression model to this manufacturing dataset."
    ),
    misleading=(
        "Use a Matern52 kernel with ARD on all inputs independently. "
        "Each feature contributes additively."
    ),
)

ARD_STRESS_PROMPTS = PromptVariants(
    dataset_name="ard_stress",
    aligned=(
        "Use a Matern52 kernel with ARD on all inputs. "
        "Only x_signal_1 and x_signal_2 are truly relevant; "
        "ARD should shrink the lengthscales of the weak and irrelevant features."
    ),
    vague=(
        "Fit a GP to this dataset."
    ),
    misleading=(
        "Use a Periodic kernel because the data has cyclical structure. "
        "Apply it to all inputs equally."
    ),
)

# ---------------------------------------------------------------------------
# Tier 1 — BoTorch test functions
# ---------------------------------------------------------------------------

BRANIN_PROMPTS = PromptVariants(
    dataset_name="branin",
    aligned=(
        "Use an RBF kernel with ARD on x0 and x1. "
        "The function is smooth with a few local optima."
    ),
    vague=(
        "Fit a GP to this 2D regression dataset."
    ),
    misleading=(
        "Use a Periodic kernel because x0 appears cyclical."
    ),
)

HARTMANN6_PROMPTS = PromptVariants(
    dataset_name="hartmann6",
    aligned=(
        "Use a Matern52 kernel with ARD on all six inputs x0 through x5. "
        "The function is smooth with a single global optimum."
    ),
    vague=(
        "Fit a standard GP to this 6-dimensional dataset."
    ),
    misleading=(
        "Use a linear kernel because the response appears to be a linear combination "
        "of the input features."
    ),
)

ROSENBROCK_PROMPTS = PromptVariants(
    dataset_name="rosenbrock",
    aligned=(
        "Use a Matern52 kernel with ARD on all four inputs. "
        "The function has a curved banana-shaped valley."
    ),
    vague=(
        "Fit a GP to this 4-dimensional regression problem."
    ),
    misleading=(
        "Use an additive kernel structure where each input contributes independently."
    ),
)

# ---------------------------------------------------------------------------
# Registry of all prompt variants
# ---------------------------------------------------------------------------

ALL_PROMPT_VARIANTS: dict[str, PromptVariants] = {
    "additive": ADDITIVE_PROMPTS,
    "periodic_decay": PERIODIC_DECAY_PROMPTS,
    "interaction": INTERACTION_PROMPTS,
    "ard_stress": ARD_STRESS_PROMPTS,
    "branin": BRANIN_PROMPTS,
    "hartmann6": HARTMANN6_PROMPTS,
    "rosenbrock": ROSENBROCK_PROMPTS,
}


def get_prompts(dataset_name: str) -> PromptVariants:
    """Retrieve prompt variants for a given dataset name.

    Args:
        dataset_name: The benchmark dataset identifier.

    Returns:
        PromptVariants for the named dataset.

    Raises:
        KeyError: If ``dataset_name`` is not in the registry.
    """
    if dataset_name not in ALL_PROMPT_VARIANTS:
        raise KeyError(
            f"No prompts defined for dataset '{dataset_name}'. "
            f"Available: {sorted(ALL_PROMPT_VARIANTS)}"
        )
    return ALL_PROMPT_VARIANTS[dataset_name]
