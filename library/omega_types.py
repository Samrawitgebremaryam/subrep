"""Result type for Omega-based skill selection — mirrors the role of
MDNRuntimeSelector's SelectionResult, but only carries fields that are
honestly true of how Omega decides (no alpha/weights_used/support_values,
since Omega doesn't infer a weight distribution the way the MDN does)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.mdn_contracts import CandidateSkillRecord


@dataclass(frozen=True)
class OmegaSelectionResult:
    """Output of one Omega-based selection step."""

    selected_skill_id: str
    selected_score: Optional[float]
    reasoning: str
    candidate_skills: tuple[CandidateSkillRecord, ...]
    context: tuple[float, ...]
    raw_response: str