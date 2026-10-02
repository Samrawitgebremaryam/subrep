"""SubRep's side of the file-based bridge to Omega's SubRepChannel plugin."""

import json
import time
import uuid
from pathlib import Path

import numpy as np

REQUESTS_DIR = Path.home() / "subrep" / "data" / "omega_requests"
RESULTS_DIR = Path.home() / "subrep" / "data" / "omega_selections"


def _format_prompt(candidates) -> str:
    lines = ["SKILL SELECTION REQUEST", "Certified candidate skills:"]
    for c in candidates:
        lines.append(f"- id={c.skill_id} delta_r={c.delta_r:.3f} delta_n={list(c.delta_n)} gate={c.gate_type}")
    ids = ", ".join(c.skill_id for c in candidates)
    lines.append(f"Reply with EXACTLY: select-skill <skill_id>, where <skill_id> is one of: {ids}")
    return "\n".join(lines)

def request_selection(observation, candidates, timeout: float = 300.0) -> str:
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    answer_path = RESULTS_DIR / "answer.json"
    if answer_path.exists():
        answer_path.unlink()  # clear any stale answer before asking again

    (REQUESTS_DIR / "request.json").write_text(json.dumps({"prompt_text": _format_prompt(candidates)}))

    elapsed = 0.0
    while not answer_path.exists():
        time.sleep(0.5)
        elapsed += 0.5
        if elapsed > timeout:
            raise TimeoutError("No response from Omega")
    return json.loads(answer_path.read_text())["selected_skill_id"]