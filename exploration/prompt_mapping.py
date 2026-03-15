from __future__ import annotations

from .env import ExplorationParams


def params_from_prompt(prompt: str) -> ExplorationParams:
    """Very simple keyword-based mapping from prompt to exploration params."""
    p = prompt.lower()

    # Defaults
    action_noise = 0.2
    step_size = 1
    curiosity_weight = 0.5

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

