"""Stable Deployment Operator authority and separately versioned reasoning policy."""

from importlib.resources import files

DEPLOYMENT_SKILL_VERSION = "deployment-v2"
DEPLOYMENT_PROMPT_VERSION = f"5c-core-v1+{DEPLOYMENT_SKILL_VERSION}"

DEPLOYMENT_CORE_PROMPT = (
    "You are the Deployment Operator, a specialist supporting Task Manager. "
    "Use only the bounded supplied forecast and reference context. "
    "Explain recorded comparisons, request typed read-only sensitivity, or propose a typed "
    "adjustment draft; never calculate "
    "forecast values, distances, or metrics. Never apply adjustments, refit models, "
    "call tools, or speak directly to the user. User confirmation is required before "
    "any numerical postprocessing."
)


def load_deployment_skill() -> str:
    policy = (
        files("interactive_forecasting.agents")
        .joinpath("skills")
        .joinpath("deployment.md")
        .read_text(encoding="utf-8")
    )
    if not policy.startswith("# Deployment skill — v2\n"):
        raise ValueError("Deployment skill version does not match runtime")
    return policy


def deployment_instructions() -> str:
    return f"{DEPLOYMENT_CORE_PROMPT}\n\n{load_deployment_skill()}"
