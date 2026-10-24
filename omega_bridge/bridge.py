"""File-based bridge to Omega. Builds the prompt from the EXACT same inputs
the MDN selector receives: the observation and the full candidate records."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path.home() / "subrep" / "data"
REQUESTS_DIR = BASE_DIR / "omega_requests"
RESULTS_DIR = BASE_DIR / "omega_selections"
LAST_REASON = ""

CANDIDATE_FIELDS = (
    "skill_id", "delta_r", "delta_n", "is_certified", "gate_type",
    "admission_margin", "epsilon", "baseline_id", "metadata",
)


def _clean(value):
    """Turn numpy numbers/tuples into plain JSON-friendly Python values."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def build_prompt(observation, candidates) -> str:
    obs = [float(v) for v in np.asarray(observation, dtype=np.float32).reshape(-1)]
    ids = [c.skill_id for c in candidates]

    lines = ["SKILL SELECTION REQUEST", f"Observation: {json.dumps(obs)}",
             "Certified candidate skills (all fields):"]
    for c in candidates:
        record = {f: _clean(getattr(c, f, None)) for f in CANDIDATE_FIELDS}
        lines.append("- " + json.dumps(record, default=str))

    
    lines.append(
        'Reply with EXACTLY one command: select-skill <skill_id> "<reason>", '
        "where <skill_id> is one of: " + ", ".join(ids) + ". "
        "The reason is your own explanation of why you chose it, in one or two plain sentences."
    )
    return "\n".join(lines)


def request_selection(observation, candidates, timeout: float = 300) -> str:
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    answer_path = RESULTS_DIR / "answer.json"
    if answer_path.exists():
        answer_path.unlink()  # clear any stale answer

    prompt = build_prompt(observation, candidates)

    # write outside the watched folder first, then move in, so Omega never
    # reads a half-written file
    tmp = BASE_DIR / "request.json.tmp"
    tmp.write_text(json.dumps({"prompt_text": prompt}))
    tmp.replace(REQUESTS_DIR / "request.json")

    deadline = time.time() + timeout
    while not answer_path.exists():
        if time.time() > deadline:
            raise TimeoutError("No response from Omega")
        time.sleep(1)

    time.sleep(0.2)  # let the write finish
    global LAST_REASON
    data = json.loads(answer_path.read_text())
    LAST_REASON = data.get("reason", "")
    return data["selected_skill_id"]
    