from __future__ import annotations

import os

from .env import ExplorationParams
from .llm_client import params_from_llm


def _llm_enabled() -> bool:
    """True if USE_LLM env is set to an affirmative value (1, true, yes)."""
    v = (os.environ.get("USE_LLM") or "").strip().lower()
    return v in ("1", "true", "yes")


def params_from_prompt(prompt: str) -> ExplorationParams:
    """
    Map a text prompt to exploration parameters.

    If USE_LLM=1 (or true/yes) and a Nemotron endpoint is configured
    (NEBIUS_LLM_ENDPOINT / NEBIUS_TOKEN), we ask the LLM to choose parameters.
    Otherwise we use the local keyword mapping only.
    """
    p = prompt.lower()

    # Defaults
    action_noise = 0.2
    step_size = 1
    curiosity_weight = 0.5

    # Try LLM only when the switch is on and endpoint/token are set
    if _llm_enabled():
        llm_cfg = params_from_llm(prompt)
        if llm_cfg is not None:
            return ExplorationParams(
                action_noise=llm_cfg.action_noise,
                step_size=llm_cfg.step_size,
                curiosity_weight=llm_cfg.curiosity_weight,
            )

    # Keyword mapping (default when USE_LLM is off or LLM call failed)
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

