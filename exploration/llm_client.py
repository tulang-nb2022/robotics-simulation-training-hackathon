from __future__ import annotations

"""
Thin wrapper around a Nemotron-3-Super-120B-a12b (or similar) endpoint hosted on Nebius.

This module is intentionally minimal and environment-driven so you can plug in
Nebius Token Factory–managed credentials without hard-coding them here.

Configuration (all via environment variables):

- NEBIUS_LLM_ENDPOINT:
    Full HTTPS URL for the Nemotron model endpoint.
- NEBIUS_TOKEN:
    Bearer token obtained from Nebius Token Factory.

The expected API shape is:

  POST $NEBIUS_LLM_ENDPOINT
  Authorization: Bearer $NEBIUS_TOKEN
  Content-Type: application/json

  {
    "model": "Nemotron-3-Super-120b-a12b",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }

and the response contains JSON with a top-level "choices[0].message.content"
string. If your actual endpoint differs, you only need to adjust
`_call_raw` accordingly.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class LLMExplorationConfig:
    action_noise: float
    step_size: int
    curiosity_weight: float


def _call_raw(system_prompt: str, user_prompt: str) -> Optional[str]:
    endpoint = os.environ.get("NEBIUS_LLM_ENDPOINT")
    token = os.environ.get("NEBIUS_TOKEN")

    if not endpoint or not token:
        return None

    payload: Dict[str, Any] = {
        "model": "Nemotron-3-Super-120b-a12b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Adjust this to match your actual API response shape
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def params_from_llm(prompt: str) -> Optional[LLMExplorationConfig]:
    """
    Ask the Nemotron model to explain and parameterize the exploration mode.

    Returns None if the LLM is not configured or if parsing fails.
    """
    system_prompt = (
        "You control a simple gridworld agent that explores a 2D grid. "
        "There are four exploration modes: cautious, random, aggressive, strange. "
        "Given a mode name, you must return JSON ONLY (no extra text) with three fields:\n"
        "{\n"
        '  "action_noise": float between 0 and 1,\n'
        '  "step_size": integer step length (1, 2, or 3),\n'
        '  "curiosity_weight": non-negative float representing how strongly the agent prefers unseen cells\n'
        "}\n"
        "Action noise is the probability of ignoring the intended direction and picking a random move. "
        "Cautious should have low noise and low curiosity. Random should have high noise. "
        "Aggressive should prefer larger steps, moving away from the start. "
        "Strange should have high curiosity, strongly preferring cells it has not visited yet.\n"
        "Again, reply with raw JSON only, no Markdown, no comments."
    )

    user_prompt = f"Mode: {prompt.strip().lower()}"

    content = _call_raw(system_prompt, user_prompt)
    if content is None:
        return None

    try:
        parsed = json.loads(content)
        return LLMExplorationConfig(
            action_noise=float(parsed["action_noise"]),
            step_size=int(parsed["step_size"]),
            curiosity_weight=float(parsed["curiosity_weight"]),
        )
    except Exception:
        return None

