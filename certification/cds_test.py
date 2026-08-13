"""
Cone-Dominant Subtask (CDS) Admission Gate.

Implements Definition 1 from SubRep Paper Section 3.2:
    Δr ≥ max_i(-Δn_i)
    
Equivalently: Δr + min_i(Δn_i) ≥ 0

CDS ensures skills are universally beneficial across all weight vectors
in the simplex cone (no coordinate "pays" more than baseline improves).
"""

from __future__ import annotations
import numpy as np
from typing import Union
from .gate import AdmissionGate, Scalar
from utils.cone_utils import compute_worst_case_motive  

class CDSGate(AdmissionGate):
    """
    Cone-Dominant Subtask admission gate.
    
    Admits skills that are universally beneficial (no motive coordinate
    worsens more than the baseline payoff improves).
    
    Reference: SubRep Paper Section 3.2, Definition 1
    """
    
    def admit(self, delta_r: Scalar, delta_n: np.ndarray, weight_set=None) -> bool:
        """
        Check if skill satisfies CDS condition.
        
        Formula: Δr + min_i(Δn_i) ≥ 0
        
        Args:
            delta_r: Scalar payoff improvement.
            delta_n: Motive improvement vector.
        
        Returns:
            True if the worst-case score over the weight set is non-negative.
        """
        self.validate_inputs(delta_r, delta_n)

        if weight_set is None or weight_set.is_empty():
            min_motive = compute_worst_case_motive(delta_n)
            return bool(float(delta_r) + min_motive >= 0.0)

        min_score = weight_set.get_worst_case_score(delta_n)
        return bool(float(delta_r) + min_score >= 0.0)
    
    def get_gate_type(self) -> str:
        """Return gate type identifier."""
        return "CDS"
    
    def get_admission_margin(self, delta_r: Scalar, delta_n: np.ndarray, weight_set=None) -> float:
        """
        Calculate how much the skill passes/fails by.
        
        Positive margin = admitted, Negative margin = rejected.
        
        Returns:
            Margin value under the active weight set.
        """
        self.validate_inputs(delta_r, delta_n)
        if weight_set is None or weight_set.is_empty():
            return float(delta_r) + float(np.min(delta_n))
        return float(delta_r) + weight_set.get_worst_case_score(delta_n)
