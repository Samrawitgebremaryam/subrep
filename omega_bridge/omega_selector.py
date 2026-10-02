# """Drop-in alternative to MDNRuntimeSelector — routes the decision through
# the actual running Omega process via the file-based bridge."""

# from __future__ import annotations

# import numpy as np

# from omega_bridge.bridge import request_selection
# from library.omega_types import OmegaSelectionResult


# class OmegaSelector:
#     def select(self, observation, candidate_skills) -> OmegaSelectionResult:
#         certified = tuple(c for c in candidate_skills if c.is_certified)
#         if not certified:
#             raise ValueError("select() requires at least one certified candidate skill")

#         obs = np.asarray(observation, dtype=np.float32).reshape(-1)
#         selected_id = request_selection(obs, certified)

#         return OmegaSelectionResult(
#             selected_skill_id=selected_id,
#             selected_score=None,
#             reasoning="(reasoning not yet captured — see open item below)",
#             candidate_skills=certified,
#             context=tuple(float(v) for v in obs),
#             raw_response="",
#         )
"""Drop-in alternative to MDNRuntimeSelector: routes the decision through
the running Omega process via the file-based bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omega_bridge.bridge import request_selection
from library.omega_types import OmegaSelectionResult
from utils.mdn_contracts import CandidateSkillRecord

LIBRARY_PATH = Path.home() / "subrep" / "data" / "library.json"


def load_candidates_from_library_json(path: Path = LIBRARY_PATH):
    data = json.loads(Path(path).read_text())
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


class OmegaSelector:
    def select(self, observation, candidate_skills) -> OmegaSelectionResult:
        certified = tuple(c for c in candidate_skills if c.is_certified)
        if not certified:
            raise ValueError("select() requires at least one certified candidate skill")

        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        selected_id = request_selection(obs, certified)

        return OmegaSelectionResult(
            selected_skill_id=selected_id,
            selected_score=None,
            reasoning="(reasoning not captured)",
            candidate_skills=certified,
            context=tuple(float(v) for v in obs),
            raw_response="",
        )

    def select_from_library(self, observation, skill_library=None) -> OmegaSelectionResult:
        """Same method name as MDNRuntimeSelector so the pipeline can swap them.
        Reads the certified skills the pipeline saved to library.json."""
        return self.select(observation, load_candidates_from_library_json())