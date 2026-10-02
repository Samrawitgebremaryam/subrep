"""Loads SubRep's real certified skills from data/library.json and sends
them to Omega for selection — replaces the earlier fake-candidate test."""

import json
from pathlib import Path

import numpy as np

from omega_bridge.bridge import REQUESTS_DIR, RESULTS_DIR, request_selection
from utils.mdn_contracts import CandidateSkillRecord

LIBRARY_PATH = Path.home() / "subrep" / "data" / "library.json"


def load_real_candidates():
    data = json.loads(LIBRARY_PATH.read_text())
    candidates = []
    for skill_id, entry in data["skills"].items():
        cert = entry["certificate"]
        candidates.append(CandidateSkillRecord(
            skill_id=skill_id,
            delta_r=cert["delta_r"],
            delta_n=tuple(cert["delta_n"]),
            is_certified=True,
            gate_type=cert["gate_type"],
            metadata={},
            admission_margin=cert["admission_margin"],
            epsilon=cert["epsilon"] if cert["epsilon"] else None,
            baseline_id=cert["baseline_id"],
        ))
    return candidates


if __name__ == "__main__":
    candidates = load_real_candidates()
    print(f"Loaded {len(candidates)} real certified skills:")
    for c in candidates:
        print(f"  {c.skill_id}: delta_r={c.delta_r:.2f}, gate={c.gate_type}")

    best = max(candidates, key=lambda c: c.delta_r)
    print(f"\nBest by delta_r alone: {best.skill_id} (delta_r={best.delta_r:.2f})")

    print("\nSending to Omega... this will block until an answer file appears.")
    print(f"Watch: {REQUESTS_DIR}")
    result = request_selection(np.zeros(8), candidates, timeout=300)
    print("\nOmega selected:", result)

    picked = next(c for c in candidates if c.skill_id == result)
    print(f"Picked skill's delta_r={picked.delta_r:.2f} (best possible was {best.delta_r:.2f})")
    print(f"Regret: {best.delta_r - picked.delta_r:.2f}")