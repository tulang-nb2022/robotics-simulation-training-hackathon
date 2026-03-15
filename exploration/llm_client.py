from __future__ import annotations

"""
Thin wrapper around a Nemotron-3-Super-120B-a12b (or similar) endpoint hosted on Nebius.

This module is intentionally minimal and environment-driven so you can plug in
Nebius Token Factory–managed credentials without hard-coding them here.

Configuration (all via environment variables):

- NEBIUS_LLM_ENDPOINT:
    Full HTTPS URL for the Nemotron model endpoint.
- NEBIUS_API_KEY:
    Bearer token obtained from Nebius Token Factory.

The expected API shape is:

  POST $NEBIUS_LLM_ENDPOINT
  Authorization: Bearer $NEBIUS_API_KEY
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


@dataclass
class LLMExplorationConfig:
    action_noise: float
    step_size: int
    curiosity_weight: float


def _debug_llm() -> bool:
    return (os.environ.get("DEBUG_LLM") or "").strip().lower() in ("1", "true", "yes")


def _call_raw(system_prompt: str, user_prompt: str) -> Optional[str]:
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if _debug_llm() or (os.environ.get("USE_LLM") or "").strip().lower() in ("1", "true", "yes"):
            print("[LLM] _call_raw: no api_key (set NEBIUS_API_KEY or OPENAI_API_KEY in .env)")
        return None

    base_url = os.environ.get("NEBIUS_LLM_ENDPOINT", "https://api.tokenfactory.us-central1.nebius.com/v1/")
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=os.environ.get("NEBIUS_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
    except Exception as e:
        if _debug_llm() or (os.environ.get("USE_LLM") or "").strip().lower() in ("1", "true", "yes"):
            print(f"[LLM] _call_raw: API call failed: {e}")
        return None

    return completion.choices[0].message.content


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
        if _debug_llm() or (os.environ.get("USE_LLM") or "").strip().lower() in ("1", "true", "yes"):
            print("[LLM] params_from_llm: _call_raw returned None (see messages above for cause)")
        return None

    try:
        parsed = json.loads(content)
    except Exception as e:
        if _debug_llm() or (os.environ.get("USE_LLM") or "").strip().lower() in ("1", "true", "yes"):
            print(f"[LLM] params_from_llm: JSON parse failed: {e!r}", "raw:", (content[:200] if content else "") + ("..." if len(content or "") > 200 else ""))
        return None
    if _debug_llm():
        print("[LLM] params_from_llm:", repr(user_prompt), "->", parsed)
    return LLMExplorationConfig(
        action_noise=float(parsed["action_noise"]),
        step_size=int(parsed["step_size"]),
        curiosity_weight=float(parsed["curiosity_weight"]),
    )

