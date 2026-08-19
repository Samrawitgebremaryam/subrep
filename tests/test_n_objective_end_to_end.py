"""End-to-end N=3-objective test for the MDN generalization work.

This deliberately exercises the whole path with num_objectives=3 (not 2),
per review feedback: record construction, candidate selection,
certification + skill-library reuse, offline training, and checkpoint
loading, all in one flow. Each step would previously fail (either with a
hard `ValueError` or a shape-mismatch) before the generalization fixes in
generator/mdn.py, utils/mdn_contracts.py, utils/mdn_selection.py,
utils/mdn_record_builder.py, generator/train_mdn.py, and
certification/certificate_schema.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from generator.mdn import MotiveDecompositionNetwork
from generator.train_mdn import train_mdn_from_records
from library.skill_library import SkillLibrary
from library.skill_metadata import MDN_WX
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.mdn_logging import build_decision_record
from utils.mdn_record_builder import build_candidate_skill_records
from utils.mdn_selection import alpha_to_mean_weights, select_best_candidate

NUM_OBJECTIVES = 3
CONTEXT_DIM = 4


def _baseline_stats() -> dict:
    return {"baseline_payoff": 0.0, "baseline_motives": (0.0,) * NUM_OBJECTIVES}


def _context(seed: float) -> tuple[float, ...]:
    return tuple(float(seed + i) for i in range(CONTEXT_DIM))


def test_n_objective_record_construction():
    """Step 1: candidate records with a 3-length delta_n build cleanly."""
    outcomes = [
        {"context": _context(0.0), "skill_id": "candidate_a", "payoff": 2.0,
         "motives": (1.0, 0.5, -0.2)},
        {"context": _context(0.0), "skill_id": "candidate_b", "payoff": 0.5,
         "motives": (0.1, 0.1, 0.1)},
    ]
    records = build_candidate_skill_records(
        skill_outcomes=outcomes, baseline_stats=_baseline_stats(),
    )
    assert len(records) == 2
    for record in records:
        assert len(record.delta_n) == NUM_OBJECTIVES


def test_n_objective_candidate_selection():
    """Step 2: MDN forward_inference + weight-based candidate selection at M=3."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=CONTEXT_DIM, num_objectives=NUM_OBJECTIVES)

    context = _context(0.0)
    outcomes = [
        {"context": context, "skill_id": "candidate_a", "payoff": 2.0, "motives": (1.0, 0.5, -0.2)},
        {"context": context, "skill_id": "candidate_b", "payoff": 0.5, "motives": (0.1, 0.1, 0.1)},
    ]
    candidates = build_candidate_skill_records(skill_outcomes=outcomes, baseline_stats=_baseline_stats())
    assert any(c.is_certified for c in candidates)

    with torch.no_grad():
        alpha, support_values = model.forward_inference(
            torch.tensor(context, dtype=torch.float32)
        )
    assert alpha.shape == (NUM_OBJECTIVES,)
    assert support_values.shape == (NUM_OBJECTIVES,)
    assert torch.all(support_values >= 0) and torch.all(support_values <= 1)
    assert torch.sum(support_values) >= 1.0 - 1e-5

    weights = alpha_to_mean_weights(alpha.numpy())
    assert weights.shape == (NUM_OBJECTIVES,)
    selected_id, selected_score = select_best_candidate(candidates, weights)
    assert selected_id in {"candidate_a", "candidate_b"}
    assert np.isfinite(selected_score)


def test_n_objective_certification_and_skill_library_reuse():
    """Step 3: CDS certification and MDN_WX skill-library admission/reuse at M=3."""
    delta_r = 1.0
    delta_n = np.array([0.6, 0.3, 0.2])  # min(delta_n) = 0.2 -> CDS passes (delta_r + 0.2 >= 0)
    gate = CDSGate()
    assert gate.admit(delta_r, delta_n)
    margin = gate.get_admission_margin(delta_r, delta_n)

    torch.manual_seed(1)
    model = MotiveDecompositionNetwork(input_dim=CONTEXT_DIM, num_objectives=NUM_OBJECTIVES)
    context = _context(1.0)
    with torch.no_grad():
        _, support_values = model.forward_inference(torch.tensor(context, dtype=torch.float32))
    support_values_tuple = tuple(float(v) for v in support_values)

    cert = Certificate(
        skill_id="skill_n3",
        gate_type="CDS",
        delta_r=delta_r,
        delta_n=tuple(float(v) for v in delta_n),
        admission_margin=margin,
        epsilon=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=1,
        gamma=0.99,
        baseline_id="idle_policy_v1",
        environment="test-n3-env",
        episode_length=50,
        version="0.1.0",
        weight_region_type=MDN_WX,
        certification_context=context,
        mdn_alpha=(1.0, 1.0, 1.0),
        wx_support_directions=tuple(tuple(row) for row in np.eye(NUM_OBJECTIVES).tolist()),
        wx_support_values=support_values_tuple,
    )
    assert len(cert.delta_n) == NUM_OBJECTIVES

    library = SkillLibrary()
    added = library.add_skill(
        "skill_n3", cert, lambda obs: 0,
        weight_region_type=MDN_WX,
        certification_context=context,
        mdn_alpha=(1.0, 1.0, 1.0),
        wx_support_directions=tuple(tuple(row) for row in np.eye(NUM_OBJECTIVES).tolist()),
        wx_support_values=support_values_tuple,
    )
    assert added, "3-objective MDN_WX skill should be admitted to the library"

    # Reuse: query admissibility under a fresh 3-length weight vector and
    # the same support geometry, with no retraining.
    current_weight = np.array([0.4, 0.3, 0.3])
    admissible = library.query_admissible(
        current_weight,
        support_directions=np.eye(NUM_OBJECTIVES),
        support_values=np.asarray(support_values_tuple),
    )
    assert any(entry.skill_id == "skill_n3" for entry in admissible)


def test_n_objective_training_and_checkpoint_loading(tmp_path: Path):
    """Steps 4 & 5: offline training at M=3, then reloading the checkpoint."""
    torch.manual_seed(2)
    seed_model = MotiveDecompositionNetwork(input_dim=CONTEXT_DIM, num_objectives=NUM_OBJECTIVES)
    context = _context(2.0)
    with torch.no_grad():
        alpha, support_values = seed_model.forward_inference(
            torch.tensor(context, dtype=torch.float32)
        )

    outcomes = [
        {"context": context, "skill_id": "candidate_a", "payoff": 2.0, "motives": (1.0, 0.5, -0.2)},
        {"context": context, "skill_id": "candidate_b", "payoff": 0.5, "motives": (0.1, 0.1, 0.1)},
    ]
    candidates = build_candidate_skill_records(skill_outcomes=outcomes, baseline_stats=_baseline_stats())
    weights = alpha_to_mean_weights(alpha.numpy())
    selected_id, selected_score = select_best_candidate(candidates, weights)

    record = build_decision_record(
        context=context,
        alpha=alpha.numpy(),
        support_values=support_values.numpy(),
        weights_used=weights,
        candidate_skills=candidates,
        selected_skill_id=selected_id,
        selected_score=selected_score,
        actual_payoff=2.0,
        actual_motives=(1.0, 0.5, -0.2),
    )
    assert len(record.alpha) == NUM_OBJECTIVES

    checkpoint_path = tmp_path / "mdn_policy_n3.pth"
    metrics = train_mdn_from_records(
        [record], checkpoint_path=str(checkpoint_path), seed=0, device="cpu",
    )
    assert Path(metrics["checkpoint_path"]).exists()

    # Step 5: checkpoint loading — shape inference must recover M=3, and
    # the reloaded model must still produce feasible support geometry.
    loaded_model = load_mdn_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded_model.num_objectives == NUM_OBJECTIVES
    assert loaded_model.input_dim == CONTEXT_DIM

    with torch.no_grad():
        loaded_alpha, loaded_support = loaded_model.forward_inference(
            torch.tensor(_context(3.0), dtype=torch.float32)
        )
    assert loaded_alpha.shape == (NUM_OBJECTIVES,)
    assert loaded_support.shape == (NUM_OBJECTIVES,)
    assert torch.all(loaded_support >= 0) and torch.all(loaded_support <= 1)
    assert torch.sum(loaded_support) >= 1.0 - 1e-5