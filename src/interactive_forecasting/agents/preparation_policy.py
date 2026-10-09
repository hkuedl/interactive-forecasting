"""Versioned Preparation guidance, separate from the stable agent role prompt."""

from __future__ import annotations

from importlib.resources import files

PREPARATION_SKILL_VERSION = "preparation-v5"
PREPARATION_PROMPT_VERSION = f"4b1-core-v1+{PREPARATION_SKILL_VERSION}"

PREPARATION_CORE_PROMPT = (
    "You are the Preparation Assistant, a specialist supporting Task Manager. "
    "Explain Preparation decisions or propose typed drafts from the supplied context. "
    "Task Manager alone speaks to the user; the application confirms decisions and changes state. "
    "Do not access files, execute services, invent numerical results, use final-test labels, "
    "or treat a proposal as user approval."
)


def load_preparation_skill() -> str:
    """Read the packaged, versioned policy; fail closed if it is unavailable."""
    policy = (
        files("interactive_forecasting.agents")
        .joinpath("skills")
        .joinpath("preparation.md")
        .read_text(encoding="utf-8")
    )
    if not policy.startswith("# Preparation skill — v5\n"):
        raise ValueError("Preparation skill version does not match the runtime")
    return policy


def preparation_instructions() -> str:
    """Compose the stable role prompt with policy for one bounded SDK invocation."""
    return f"{PREPARATION_CORE_PROMPT}\n\n{load_preparation_skill()}"
