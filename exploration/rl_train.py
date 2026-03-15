from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from .env import GridWorld


PROMPTS = ["cautious", "random", "aggressive", "strange"]

Action = int  # 0:right, 1:left, 2:up, 3:down
State = Tuple[int, int]


@dataclass
class RLParams:
    episodes: int
    max_steps_per_episode: int
    alpha: float  # learning rate
    gamma: float  # discount
    epsilon_start: float
    epsilon_end: float


def rl_params_from_prompt(prompt: str) -> RLParams:
    """Map prompt to RL hyperparameters and implicit reward style."""
    p = prompt.lower()

    # sensible defaults
    params = RLParams(
        episodes=400,
        max_steps_per_episode=200,
        alpha=0.3,
        gamma=0.95,
        epsilon_start=0.9,
        epsilon_end=0.05,
    )

    if "cautious" in p:
        params.episodes = 400
        params.max_steps_per_episode = 120
        params.alpha = 0.2
        params.gamma = 0.99
        params.epsilon_start = 0.3
        params.epsilon_end = 0.01
    elif "random" in p:
        params.episodes = 200
        params.max_steps_per_episode = 150
        params.alpha = 0.3
        params.gamma = 0.9
        params.epsilon_start = 0.9
        params.epsilon_end = 0.4
    elif "aggressive" in p:
        params.episodes = 300
        params.max_steps_per_episode = 200
        params.alpha = 0.4
        params.gamma = 0.9
        params.epsilon_start = 0.7
        params.epsilon_end = 0.1
    elif "strange" in p:
        params.episodes = 400
        params.max_steps_per_episode = 200
        params.alpha = 0.3
        params.gamma = 0.95
        params.epsilon_start = 0.8
        params.epsilon_end = 0.05

    return params


def make_reward_fn(prompt: str, width: int, height: int):
    """Prompt-dependent reward shaping.

    - cautious: penalize revisits, encourage staying near center.
    - random: small shaping, mostly neutral.
    - aggressive: reward moving outward from center quickly.
    - strange: big bonus for visiting new cells, light penalty for revisits.
    """
    p = prompt.lower()
    cx, cy = width / 2.0, height / 2.0

    if "cautious" in p:
        def reward_fn(pos: State, visits: np.ndarray) -> float:
            x, y = pos
            r = 0.0
            # penalize revisiting
            if visits[y, x] > 1:
                r -= 0.2
            # small penalty for being far from center
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r -= 0.01 * dist
            return r

    elif "random" in p:
        def reward_fn(pos: State, visits: np.ndarray) -> float:
            x, y = pos
            # tiny novelty bonus, otherwise neutral
            return 0.02 if visits[y, x] == 1 else 0.0

    elif "aggressive" in p:
        def reward_fn(pos: State, visits: np.ndarray) -> float:
            x, y = pos
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r = 0.03 * dist
            # small penalty for revisiting
            if visits[y, x] > 1:
                r -= 0.05
            return r

    else:  # strange
        def reward_fn(pos: State, visits: np.ndarray) -> float:
            x, y = pos
            if visits[y, x] == 1:
                return 0.5  # big novelty reward
            # small random-ish penalty/bonus to create odd attractors
            noise = (hash((x, y)) % 5 - 2) * 0.02
            return -0.05 + noise

    return reward_fn


def epsilon_for_episode(params: RLParams, episode_idx: int) -> float:
    frac = episode_idx / max(params.episodes - 1, 1)
    return params.epsilon_start + frac * (params.epsilon_end - params.epsilon_start)


def choose_action(q: np.ndarray, state: State, eps: float, rng: np.random.Generator) -> Action:
    if rng.random() < eps:
        return int(rng.integers(0, 4))
    x, y = state
    return int(np.argmax(q[y, x]))


def apply_action(pos: State, action: Action, width: int, height: int) -> State:
    x, y = pos
    if action == 0:   # right
        x = min(x + 1, width - 1)
    elif action == 1:  # left
        x = max(x - 1, 0)
    elif action == 2:  # up
        y = min(y + 1, height - 1)
    elif action == 3:  # down
        y = max(y - 1, 0)
    return x, y


def train_q_learning(prompt: str, seed: int | None = None) -> np.ndarray:
    """Train a tabular Q-learning agent whose behavior is shaped by the prompt.

    Returns a visit heatmap created by executing the greedy policy once after training.
    """
    params = rl_params_from_prompt(prompt)

    # We use GridWorld only as a convenient container for visits and size.
    env = GridWorld(seed=seed)
    width, height = env.width, env.height

    q = np.zeros((height, width, 4), dtype=np.float32)  # (y, x, action)
    rng = np.random.default_rng(seed)

    reward_fn = make_reward_fn(prompt, width, height)

    for ep in range(params.episodes):
        env.reset()
        state: State = (env.agent_pos[0].item(), env.agent_pos[1].item())
        eps = epsilon_for_episode(params, ep)

        for _ in range(params.max_steps_per_episode):
            action = choose_action(q, state, eps, rng)
            next_state = apply_action(state, action, width, height)

            # apply move in env, update visits
            env.agent_pos[0] = next_state[0]
            env.agent_pos[1] = next_state[1]
            env._visit_current()

            reward = reward_fn(next_state, env.visits)

            sx, sy = state
            nx, ny = next_state

            best_next = float(np.max(q[ny, nx]))
            td_target = reward + params.gamma * best_next
            td_error = td_target - float(q[sy, sx, action])
            q[sy, sx, action] += params.alpha * td_error

            state = next_state

    # After training, run one long greedy rollout to visualize policy.
    env.reset()
    state = (env.agent_pos[0].item(), env.agent_pos[1].item())
    for _ in range(params.max_steps_per_episode * 2):
        action = choose_action(q, state, eps=0.0, rng=rng)
        next_state = apply_action(state, action, width, height)
        env.agent_pos[0] = next_state[0]
        env.agent_pos[1] = next_state[1]
        env._visit_current()
        state = next_state

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
    base_dir = Path(os.environ.get("EXPLORATION_OUTPUT_DIR", "outputs"))
    output_dir = base_dir / "rl"

    for prompt in PROMPTS:
        visits = train_q_learning(prompt, seed=42)
        out_file = output_dir / f"rl_exploration_{prompt}.png"
        save_heatmap(visits, out_file, title=f"RL exploration pattern: {prompt}")
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()

