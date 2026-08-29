"""Support-function geometry utilities for context-conditioned weight sets W_x.
"""

from __future__ import annotations

import numpy as np


def make_basis_query_directions(num_objectives: int) -> np.ndarray:
    """Return standard basis query directions for the objective space.
    """
    if num_objectives <= 0:
        raise ValueError(f"num_objectives must be positive, got {num_objectives}")
    return np.eye(num_objectives, dtype=np.float32)


def compute_support_values_from_vertices(vertices: np.ndarray, query_directions: np.ndarray) -> np.ndarray:
    """Compute support-function values h_W(u_j) = max_{w in W} u_j^T w.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    query_directions = np.asarray(query_directions, dtype=np.float32)

    if vertices.ndim != 2:
        raise ValueError(f"vertices must have shape (N, M), got {vertices.shape}")
    if query_directions.ndim != 2:
        raise ValueError(f"query_directions must have shape (K, M), got {query_directions.shape}")
    if vertices.shape[0] == 0:
        raise ValueError("vertices must contain at least one weight vector")
    if vertices.shape[1] != query_directions.shape[1]:
        raise ValueError(
            f"vertices dimension {vertices.shape[1]} must match query direction dimension {query_directions.shape[1]}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values")
    if not np.all(np.isfinite(query_directions)):
        raise ValueError("query_directions must contain only finite values")

    scores = query_directions @ vertices.T
    return np.max(scores, axis=1).astype(np.float32)


def simplex_support_values(query_directions: np.ndarray) -> np.ndarray:
    """Compute support-function values for the full simplex.
    """
    query_directions = np.asarray(query_directions, dtype=np.float32)
    if query_directions.ndim != 2:
        raise ValueError(f"query_directions must have shape (K, M), got {query_directions.shape}")
    if query_directions.shape[0] == 0 or query_directions.shape[1] == 0:
        raise ValueError("query_directions must be non-empty")
    if not np.all(np.isfinite(query_directions)):
        raise ValueError("query_directions must contain only finite values")
    return np.max(query_directions, axis=1).astype(np.float32)


def validate_box_support_values(support_values: np.ndarray) -> np.ndarray:
    """Validate M-objective W_x support values on the standard basis directions.

    support_values[i] is h_Wx(e_i) = max_{w in Wx} w_i, i.e. an upper bound
    on how much weight objective i may receive inside Wx:

        Wx = {w : w >= 0, sum(w) == 1, w_i <= support_values[i] for all i}

    Well-defined (non-empty) for any M >= 1 as long as every value lies in
    [0, 1] and the values sum to at least 1. Works for any M.
    """
    support_values = np.asarray(support_values, dtype=np.float64).reshape(-1)
    if support_values.shape[0] == 0:
        raise ValueError("support_values must be non-empty")
    if not np.all(np.isfinite(support_values)):
        raise ValueError("support_values must contain only finite values")
    if not np.all((support_values >= 0.0) & (support_values <= 1.0)):
        raise ValueError(f"support_values must satisfy 0 <= s_i <= 1, got {support_values.tolist()}")
    if float(np.sum(support_values)) < 1.0 - 1e-6:
        raise ValueError(
            "support_values must satisfy sum(s_i) >= 1 (otherwise Wx is empty), "
            f"got sum={float(np.sum(support_values)):.6f} from {support_values.tolist()}"
        )
    return support_values


def box_simplex_worst_case_score(support_values: np.ndarray, direction: np.ndarray) -> float:
    """Compute min_{w in Wx} w . direction for the box-capped simplex Wx.

    Solved via greedy/water-filling: allocate as much of the unit weight
    budget as possible to the objectives with the smallest direction value
    first, up to each objective's cap. Exact for any M >= 1.
    """
    support_values = validate_box_support_values(support_values)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    if direction.shape != support_values.shape:
        raise ValueError(
            f"direction shape {direction.shape} must match support_values shape {support_values.shape}"
        )
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must contain only finite values")

    order = np.argsort(direction, kind="stable")
    remaining = 1.0
    total = 0.0
    for index in order:
        if remaining <= 1e-12:
            break
        take = min(float(support_values[index]), remaining)
        total += take * float(direction[index])
        remaining -= take
    return float(total)
