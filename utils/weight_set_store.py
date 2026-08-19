"""Persistent monotone weight-set store for context-conditioned W_x tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json

import numpy as np

from utils.support_geometry import (
    box_simplex_worst_case_score,
    compute_support_values_from_vertices,
    make_basis_query_directions,
    simplex_support_values,
    validate_box_support_values,
)
@dataclass
class WeightSet:
    """Weight set W_x for a single context.

    Can be represented either as a finite set of observed vertices, or as
    a box-capped simplex defined by per-objective support values (works
    for any number of objectives M). Only one representation is used at a
    time; box_upper_bounds takes precedence when both are supplied.
    """

    vertices: list[np.ndarray] = field(default_factory=list)
    box_upper_bounds: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return len(self.vertices) == 0 and self.box_upper_bounds is None

    def add_vertex(self, weight_vector: np.ndarray) -> None:
        weight_vector = np.asarray(weight_vector, dtype=np.float32).reshape(-1)
        if weight_vector.ndim != 1 or len(weight_vector) == 0:
            raise ValueError(f"weight_vector must be a non-empty 1D vector, got {weight_vector.shape}")
        if not np.all(np.isfinite(weight_vector)):
            raise ValueError("weight_vector must contain only finite values")
        self.vertices.append(weight_vector.copy())

    @classmethod
    def from_box_support(cls, support_values: np.ndarray) -> "WeightSet":
        """Build a WeightSet from MDN-predicted support values (any M >= 1)."""
        validated = validate_box_support_values(support_values)
        return cls(box_upper_bounds=validated.astype(np.float32))

    def get_support_values(self, query_directions: np.ndarray) -> np.ndarray:
        if self.box_upper_bounds is not None:
            query_directions = np.asarray(query_directions, dtype=np.float64)
            if query_directions.ndim != 2:
                raise ValueError(
                    f"query_directions must have shape (K, M), got {query_directions.shape}"
                )
            # h_Wx(direction) = max_{w in Wx} w . direction
            #                 = -min_{w in Wx} w . (-direction)
            # Correct for ANY query direction, not just the standard basis
            # (for a standard basis row e_i this reduces to exactly
            # box_upper_bounds[i], which is why the old shortcut of just
            # returning box_upper_bounds happened to work for the only
            # caller in this codebase — but was wrong for any other direction).
            return np.array(
                [
                    -box_simplex_worst_case_score(self.box_upper_bounds, -direction)
                    for direction in query_directions
                ],
                dtype=np.float32,
            )
        
        if self.is_empty():
            return simplex_support_values(query_directions)
        vertices_array = np.stack(self.vertices, axis=0)
        return compute_support_values_from_vertices(vertices_array, query_directions)

    def get_vertices_array(self) -> Optional[np.ndarray]:
        if len(self.vertices) == 0:
            return None
        return np.stack(self.vertices, axis=0)

    def get_worst_case_score(self, direction: np.ndarray) -> float:
        """Return min_{w in W} w . direction. Works for any M >= 1, whether
        this WeightSet is vertex-backed or box-backed."""
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if self.box_upper_bounds is not None:
            return box_simplex_worst_case_score(self.box_upper_bounds, direction)
        vertices = self.get_vertices_array()
        if vertices is None:
            return float(np.min(direction))
        return float(np.min(np.asarray(vertices, dtype=np.float64) @ direction))
    
class WeightSetStore:
    """Per-context registry of learned weight sets W_x."""

    def __init__(self, num_objectives: int) -> None:
        if num_objectives <= 0:
            raise ValueError(f"num_objectives must be positive, got {num_objectives}")
        self.num_objectives = int(num_objectives)
        self._store: dict[tuple[float, ...], WeightSet] = {}
        self._query_directions = make_basis_query_directions(num_objectives)

    def _context_key(self, context: np.ndarray) -> tuple[float, ...]:
        context = np.asarray(context, dtype=np.float32).reshape(-1)
        if context.ndim != 1 or len(context) == 0:
            raise ValueError(f"context must be a non-empty 1D vector, got {context.shape}")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")
        return tuple(np.round(context, decimals=4).tolist())

    def observe_certified_weight(self, context: np.ndarray, weight_vector: np.ndarray) -> None:
        key = self._context_key(context)
        if key not in self._store:
            self._store[key] = WeightSet()
        self._store[key].add_vertex(weight_vector)

    def get_support_values(self, context: np.ndarray) -> np.ndarray:
        key = self._context_key(context)
        weight_set = self._store.get(key, WeightSet())
        return weight_set.get_support_values(self._query_directions)

    def get_weight_set(self, context: np.ndarray) -> Optional[WeightSet]:
        """Get the WeightSet for a context, or None if not yet observed."""
        key = self._context_key(context)
        return self._store.get(key)

    def get_all_support_targets(self) -> list[tuple[np.ndarray, np.ndarray]]:
        targets: list[tuple[np.ndarray, np.ndarray]] = []
        for key, weight_set in self._store.items():
            context = np.array(key, dtype=np.float32)
            support_values = weight_set.get_support_values(self._query_directions)
            targets.append((context, support_values))
        return targets

    def context_count(self) -> int:
        return len(self._store)

    def total_vertex_count(self) -> int:
        return sum(len(weight_set.vertices) for weight_set in self._store.values())

    def save(self, path: str | Path) -> None:
        """Persist the current `W_x` store to JSON."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_objectives": self.num_objectives,
            "contexts": {
                ",".join(map(str, key)): [vertex.tolist() for vertex in weight_set.vertices]
                for key, weight_set in self._store.items()
            },
        }
        file_path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WeightSetStore":
        """Restore a `WeightSetStore` from JSON persistence."""
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        store = cls(num_objectives=int(data["num_objectives"]))
        for key_str, vertices_list in data["contexts"].items():
            key = tuple(float(value) for value in key_str.split(",") if value != "")
            weight_set = WeightSet()
            for vertex in vertices_list:
                weight_set.add_vertex(np.asarray(vertex, dtype=np.float32))
            store._store[key] = weight_set
        return store
