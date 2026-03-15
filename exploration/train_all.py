from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pygame
import matplotlib.pyplot as plt

from .env import GridWorld
from .prompt_mapping import params_from_prompt


PROMPTS = ["cautious", "random", "aggressive", "strange"]


def run_episode(prompt: str, steps: int = 500, seed: int | None = None) -> np.ndarray:
    params = params_from_prompt(prompt)
    env = GridWorld(params=params, seed=seed)

    # Base direction: move to the right; behavior will be shaped by params
    base_direction = np.array([1.0, 0.0], dtype=np.float32)

    for _ in range(steps):
        env.step(base_direction)

    # Return the visit heatmap
    return env.visits.copy()


def save_heatmap(visits: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(visits, cmap="magma", origin="lower")
    plt.colorbar(label="visit count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    pygame.display.init()
    pygame.display.set_mode((1, 1))  # headless-friendly setup

    output_dir = Path(os.environ.get("EXPLORATION_OUTPUT_DIR", "outputs"))

    for prompt in PROMPTS:
        visits = run_episode(prompt, steps=800, seed=42)
        out_file = output_dir / f"exploration_{prompt}.png"
        save_heatmap(visits, out_file, title=f"Exploration pattern: {prompt}")
        print(f"Saved {out_file}")

    pygame.quit()


if __name__ == "__main__":
    main()

