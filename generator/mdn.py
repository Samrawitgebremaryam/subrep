# Motive Decomposition Network (MDN)

from __future__ import annotations

import torch
from torch import Tensor, nn


class MotiveDecompositionNetwork(nn.Module):
    """Shared-backbone MDN for runtime inference and auxiliary training.
    """
    def __init__(
        self,
        input_dim: int = 8,
        num_objectives: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        alpha_epsilon: float = 1e-6,
        num_skills: int = 128,
        skill_embedding_dim: int = 8,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError(
                f"Expected num_hidden_layers >= 1, got {num_hidden_layers}"
            )
        if num_skills <= 0:
            raise ValueError(f"Expected num_skills > 0, got {num_skills}")

        self.input_dim = input_dim
        self.num_objectives = num_objectives
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.alpha_epsilon = alpha_epsilon
        self.num_skills = num_skills
        self.skill_embedding_dim = skill_embedding_dim

        trunk_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            trunk_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )

        self.trunk = nn.Sequential(*trunk_layers)
        self.distribution_head = nn.Linear(hidden_dim, num_objectives)
        self.support_head = nn.Linear(hidden_dim, num_objectives)  # see _support_values_from_raw
        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
        self.auxiliary_fusion = nn.Sequential(
            nn.Linear(hidden_dim + skill_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.motive_head = nn.Linear(hidden_dim, num_objectives)
        self.softplus = nn.Softplus()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply stable initialization across all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.skill_embedding.weight, mean=0.0, std=0.02)

    def _encode_context(self, context: Tensor) -> tuple[Tensor, bool]:
        if context.ndim not in (1, 2):
            raise ValueError(
                f"Expected context with shape ({self.input_dim},) or "
                f"(N, {self.input_dim}), got tensor with shape {tuple(context.shape)}"
            )

        is_single_input = context.ndim == 1
        if is_single_input:
            if context.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected single context shape ({self.input_dim},), "
                    f"got {tuple(context.shape)}"
                )
            context = context.unsqueeze(0)
        elif context.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected batched context shape (N, {self.input_dim}), "
                f"got {tuple(context.shape)}"
            )

        features = self.trunk(context)
        return features, is_single_input

    # Bump this whenever _support_values_from_raw's semantics change in a
    # way that alters outputs for the same raw_support tensor. Checkpoints
    # remain tensor-SHAPE compatible across versions, but are NOT
    # numerically/behavior compatible unless the version matches -- see
    # utils/mdn_checkpoint_loader.py, which warns loudly on a
    # mismatch/missing value instead of silently reinterpreting old
    # weights under new semantics.
    SUPPORT_PARAM_VERSION = "sigmoid_deficit_v1"

    def _support_values_from_raw(self, raw_support: Tensor) -> Tensor:
        """Map raw support-head output to valid W_x support values.

        Guarantees, for ANY num_objectives >= 1 (not just 2):
          - 0 <= s_i <= 1 for every objective i
          - sum_i s_i >= 1, so the induced box-capped-simplex region
            W_x = {w : w >= 0, sum(w) == 1, w_i <= s_i} is always non-empty

        Construction:

            base    = sigmoid(raw_support)                     # (..., M), each in (0, 1)
            deficit = relu(1 - sum(base))                       # how far short of 1 the sum is
            headroom = 1 - base                                 # remaining room before hitting the cap of 1
            boost   = deficit * headroom / sum(headroom)         # redistribute the deficit, capped-safe
            s       = base + boost

        Whenever sum(base) >= 1 already, s == base exactly (no
        redistribution). Since base = sigmoid(raw_support) is surjective
        onto (0, 1)^M coordinate-by-coordinate, EVERY feasible target
        vector with sum(s) >= 1 is exactly reachable this way (e.g.
        [0.8, 0.5, 0.1] for M=3). Needs exactly num_objectives raw
        inputs, matching the model's original (pre-generalization) tensor
        shapes -- but not its numeric behavior; see SUPPORT_PARAM_VERSION.
        """
        base = torch.sigmoid(raw_support)
        deficit = torch.clamp(1.0 - base.sum(dim=-1, keepdim=True), min=0.0)
        headroom = 1.0 - base
        headroom_sum = headroom.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        boost = deficit * headroom / headroom_sum
        return base + boost

    def forward_inference(self, context: Tensor) -> tuple[Tensor, Tensor]:
        features, is_single_input = self._encode_context(context)
        weight_params = self.softplus(self.distribution_head(features)) + self.alpha_epsilon
        support_values = self._support_values_from_raw(self.support_head(features))

        if is_single_input:
            weight_params = weight_params.squeeze(0)
            support_values = support_values.squeeze(0)

        return weight_params, support_values

    def forward_auxiliary(self, context: Tensor, skill_id: Tensor) -> tuple[Tensor, Tensor]:
        features, is_single_input = self._encode_context(context)

        if is_single_input:
            if skill_id.ndim != 0:
                raise ValueError(f"Expected scalar skill_id for single context, got shape {tuple(skill_id.shape)}")
            skill_id = skill_id.unsqueeze(0)
        else:
            if skill_id.ndim != 1:
                raise ValueError(f"Expected batched skill_id shape (N,), got shape {tuple(skill_id.shape)}")
            if skill_id.shape[0] != features.shape[0]:
                raise ValueError("skill_id batch size must match context batch size")

        embedded_skill = self.skill_embedding(skill_id.long())
        fused = self.auxiliary_fusion(torch.cat([features, embedded_skill], dim=-1))
        gate_logits = self.gate_head(fused).squeeze(-1)
        q_hat = self.motive_head(fused)

        if is_single_input:
            gate_logits = gate_logits.squeeze(0)
            q_hat = q_hat.squeeze(0)

        return gate_logits, q_hat

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_inference(context)
