"""Shared checkpoint loading helpers for MotiveDecompositionNetwork."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from generator.mdn import MotiveDecompositionNetwork


def load_mdn_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> MotiveDecompositionNetwork:
    """Load an MDN checkpoint while inferring the saved architecture shape.

    Candidate-set training may use a large skill embedding table, so runtime
    loading must reconstruct the model from checkpoint weights instead of
    assuming default constructor values.

    Tensor SHAPES have been stable across support-value formula changes
    (support_head's output size has always been num_objectives), so a
    checkpoint saved under an older formula still loads without a shape
    mismatch. It is NOT numerically/behavior compatible, though -- the same
    raw weights are interpreted differently by different formula versions.
    See _warn_on_support_param_version_mismatch below.
    """
    checkpoint = _load_checkpoint_payload(path, map_location=map_location)
    state = extract_model_state_dict(checkpoint)
    model = build_mdn_from_state_dict(state)
    _warn_on_support_param_version_mismatch(checkpoint, path)
    model.load_state_dict(state)
    model.to(torch.device(map_location))
    model.eval()
    return model


def _warn_on_support_param_version_mismatch(checkpoint: Any, path: str | Path) -> None:
    """Warn loudly when a checkpoint's support-value formula version is
    missing or does not match the currently running formula.

    This does not (and cannot) reinterpret old weights under new semantics --
    it only makes the incompatibility visible instead of silent. Retraining
    is the recommended fix, not a legacy numerical path, since keeping the
    old per-M formula alive would reintroduce the exact per-M special-casing
    this generalization removed.
    """
    current_version = MotiveDecompositionNetwork.SUPPORT_PARAM_VERSION
    saved_version = checkpoint.get("support_param_version") if isinstance(checkpoint, dict) else None

    if saved_version is None:
        warnings.warn(
            f"MDN checkpoint at {path!r} has no recorded support_param_version "
            "(it predates version tracking). It was very likely trained under a "
            "different support-value formula than the one currently running "
            f"({current_version!r}). The checkpoint's tensor shapes will load "
            "without error, but its predicted support geometry will NOT match "
            "what it was trained to produce -- retrain rather than trust its "
            "support-head outputs as-is.",
            stacklevel=3,
        )
    elif saved_version != current_version:
        warnings.warn(
            f"MDN checkpoint at {path!r} was trained under support_param_version "
            f"{saved_version!r}, but the currently running formula is "
            f"{current_version!r}. The checkpoint's tensor shapes will load "
            "without error, but its predicted support geometry will NOT match "
            "what it was trained to produce -- retrain rather than trust its "
            "support-head outputs as-is.",
            stacklevel=3,
        )


def extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the model state dict from raw or wrapped checkpoint payloads."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError("MDN checkpoint must be a state_dict or contain model_state_dict")
    return state


def build_mdn_from_state_dict(
    state: dict[str, torch.Tensor],
) -> MotiveDecompositionNetwork:
    """Instantiate MotiveDecompositionNetwork from state-dict tensor shapes."""
    _require_key(state, "trunk.0.weight")
    _require_key(state, "distribution_head.weight")

    first_trunk_weight = state["trunk.0.weight"]
    distribution_weight = state["distribution_head.weight"]

    input_dim = int(first_trunk_weight.shape[1])
    hidden_dim = int(first_trunk_weight.shape[0])
    num_hidden_layers = sum(
        1 for key in state if key.startswith("trunk.") and key.endswith(".weight")
    )
    num_objectives = int(distribution_weight.shape[0])

    skill_embedding = state.get("skill_embedding.weight")
    if skill_embedding is None:
        num_skills = 128
        skill_embedding_dim = 8
    else:
        num_skills = int(skill_embedding.shape[0])
        skill_embedding_dim = int(skill_embedding.shape[1])

    return MotiveDecompositionNetwork(
        input_dim=input_dim,
        num_objectives=num_objectives,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_skills=num_skills,
        skill_embedding_dim=skill_embedding_dim,
    )


def _load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device,
) -> Any:
    return torch.load(path, map_location=map_location)


def _require_key(state: dict[str, torch.Tensor], key: str) -> None:
    if key not in state:
        raise ValueError(f"MDN checkpoint is missing required tensor {key!r}")
