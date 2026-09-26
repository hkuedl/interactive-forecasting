"""Stable Model Manager role prompt and separately versioned strategy guidance."""

from __future__ import annotations

from importlib.resources import files

MODEL_MANAGER_SKILL_VERSION = "model-manager-v1"
MODEL_MANAGER_PROMPT_VERSION = "4c-core-v1+model-manager-v1"

MODEL_MANAGER_CORE_PROMPT = (
    "You are Model Manager, a search-strategy specialist supporting Task Manager. "
    "Analyze bounded validation-only summaries and propose typed search guidance. "
    "Do not train models, calculate authoritative metrics, modify backend state, access "
    "final-test data, or address the user directly. The application validates and applies actions."
)

MODEL_DEVELOPER_CORE_PROMPT = (
    "You are Model Developer, an execution specialist supporting Task Manager. "
    "Return only the approval ID of the application-approved request you received. "
    "Do not select search strategy, change ranges, calculate metrics, access final-test data, "
    "or claim an execution result before the deterministic service returns."
)


def load_model_manager_skill() -> str:
    policy = (
        files("interactive_forecasting.agents")
        .joinpath("skills")
        .joinpath("model_manager_v1.md")
        .read_text(encoding="utf-8")
    )
    if not policy.startswith("# Model Manager search-guidance skill — v1\n"):
        raise ValueError("Model Manager skill version does not match the runtime")
    return policy


def model_manager_instructions() -> str:
    return f"{MODEL_MANAGER_CORE_PROMPT}\n\n{load_model_manager_skill()}"
