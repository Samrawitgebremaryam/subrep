"""Integrated version: Omega selects the skill instead of the MDN.
Mirrors Phase 5 of demo/run_full_pipeline.py without modifying it."""

import time
import numpy as np

from omega_bridge.omega_selector import OmegaSelector, load_candidates_from_library_json

eval_observations = [
    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    np.array([-0.1, 0.5, -0.2, 0.3, 0.1, -0.3, 0.4, 0.2], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
]

candidates = load_candidates_from_library_json()
delta_r_by_id = {c.skill_id: c.delta_r for c in candidates}
best_id = max(delta_r_by_id, key=delta_r_by_id.get)
best_dr = delta_r_by_id[best_id]

print("=" * 60)
print("  Skill Selection Demo (selector = OMEGA)")
print("=" * 60)
print(f"[Ref] {len(candidates)} certified skills, best by delta_r: {best_id} ({best_dr:.2f})")

selector = OmegaSelector()

for idx, obs in enumerate(eval_observations, 1):
    t0 = time.perf_counter()
    try:
        result = selector.select(obs, candidates)
        elapsed = time.perf_counter() - t0
        dr = delta_r_by_id.get(result.selected_skill_id)
        print(f"  Obs {idx}: Selected skill '{result.selected_skill_id}' "
              f"(delta_r={dr}, regret={best_dr - dr:.2f}, time={elapsed:.1f}s)")
    except Exception as e:
        print(f"  Obs {idx}: Selection failed - {e}")

print("\n[OMEGA] Selection demo complete")