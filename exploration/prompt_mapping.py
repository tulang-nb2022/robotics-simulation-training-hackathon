from __future__ import annotations

from .env import ExplorationParams
from .llm_client import params_from_llm


def params_from_prompt(prompt: str) -> ExplorationParams:
    """
    Map a text prompt to exploration parameters.

    If a Nemotron endpoint is configured (via NEBIUS_LLM_ENDPOINT / NEBIUS_TOKEN),
    we first ask it to choose parameters for the mode. If that fails, we fall
    back to the local keyword mapping.
    """
    p = prompt.lower()

    # Defaults
    action_noise = 0.2
    step_size = 1
    curiosity_weight = 0.5

    # Try LLM first (if configured)
    llm_cfg = params_from_llm(prompt)
    if llm_cfg is not None:
        return ExplorationParams(
            action_noise=llm_cfg.action_noise,
            step_size=llm_cfg.step_size,
            curiosity_weight=llm_cfg.curiosity_weight,
        )

    # Fallback: simple keyword mapping
    if "cautious" in p:
        action_noise = 0.05
        step_size = 1
        curiosity_weight = 0.2
    elif "random" in p:
        action_noise = 0.6
        step_size = 1
        curiosity_weight = 0.3
    elif "aggressive" in p:
        action_noise = 0.4
        step_size = 2
        curiosity_weight = 0.1
    elif "strange" in p:
        action_noise = 0.5
        step_size = 1
        curiosity_weight = 2.0

    return ExplorationParams(
        action_noise=action_noise,
        step_size=step_size,
        curiosity_weight=curiosity_weight,
    )

