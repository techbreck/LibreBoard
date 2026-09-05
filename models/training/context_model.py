# SPDX-License-Identifier: GPL-3.0-only
"""LibreBoard's compact bilingual candidate-ranking Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class RmsNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized * self.weight


class RotarySelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.width = config["width"]
        self.heads = config["heads"]
        self.head_width = self.width // self.heads
        self.scale = self.head_width ** -0.5
        self.query = nn.Linear(self.width, self.width, bias=False)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)
        self.dropout = nn.Dropout(config["dropout"])

        half_width = self.head_width // 2
        inverse_frequency = 1.0 / (
            config["ropeTheta"] ** (torch.arange(0, half_width, dtype=torch.float32) * 2 / self.head_width)
        )
        positions = torch.arange(config["sequenceLength"], dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config["sequenceLength"], config["sequenceLength"], dtype=torch.bool)),
            persistent=False,
        )

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _width = values.shape
        return values.reshape(batch, sequence, self.heads, self.head_width).transpose(1, 2)

    def _rotary(self, values: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        cosine = self.rope_cos[position_offset: position_offset + values.shape[-2]].reshape(
            1, 1, values.shape[-2], -1
        )
        sine = self.rope_sin[position_offset: position_offset + values.shape[-2]].reshape(
            1, 1, values.shape[-2], -1
        )
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)

    def _project_pair(
        self,
        projection: nn.Linear,
        prefix: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_elements = prefix.shape[0] * prefix.shape[1]
        combined = torch.cat((
            prefix.reshape(-1, self.width),
            candidates.reshape(-1, self.width),
        ), dim=0)
        projected = projection(combined)
        prefix_output = projected[:prefix_elements].reshape(prefix.shape[0], prefix.shape[1], self.width)
        candidate_output = projected[prefix_elements:].reshape(
            candidates.shape[0], candidates.shape[1], self.width
        )
        return prefix_output, candidate_output

    def forward_pair(
        self,
        prefix: torch.Tensor,
        candidates: torch.Tensor,
        prefix_mask: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_query, candidate_query = self._project_pair(self.query, prefix, candidates)
        prefix_key, candidate_key = self._project_pair(self.key, prefix, candidates)
        prefix_value, candidate_value = self._project_pair(self.value, prefix, candidates)
        prefix_query = self._rotary(self._split_heads(prefix_query))
        candidate_query = self._rotary(self._split_heads(candidate_query), position_offset)
        prefix_key = self._rotary(self._split_heads(prefix_key))
        candidate_key = self._rotary(self._split_heads(candidate_key), position_offset)
        prefix_value = self._split_heads(prefix_value)
        candidate_value = self._split_heads(candidate_value)

        prefix_scores = torch.matmul(prefix_query, prefix_key.transpose(-1, -2)) * self.scale
        prefix_length = prefix.shape[1]
        prefix_allowed = self.causal_mask[:prefix_length, :prefix_length].reshape(
            1, 1, prefix_length, prefix_length
        )
        prefix_allowed = prefix_allowed & prefix_mask.to(dtype=torch.bool).reshape(1, 1, 1, prefix_length)
        prefix_scores = prefix_scores.masked_fill(~prefix_allowed, torch.finfo(prefix_scores.dtype).min)
        prefix_probabilities = self.dropout(functional.softmax(prefix_scores, dim=-1))
        prefix_attended = torch.matmul(prefix_probabilities, prefix_value)
        prefix_attended = prefix_attended.transpose(1, 2).contiguous().reshape(
            prefix.shape[0], prefix_length, self.width
        )

        batch = candidates.shape[0]
        candidate_length = candidates.shape[1]
        prefix_key = prefix_key.expand(batch, -1, -1, -1)
        prefix_value = prefix_value.expand(batch, -1, -1, -1)
        key = torch.cat((prefix_key, candidate_key), dim=-2)
        value = torch.cat((prefix_value, candidate_value), dim=-2)

        candidate_prefix_allowed = prefix_mask.to(dtype=torch.bool).reshape(1, 1, 1, prefix_length)
        candidate_prefix_allowed = candidate_prefix_allowed.expand(batch, 1, candidate_length, prefix_length)
        candidate_allowed = self.causal_mask[:candidate_length, :candidate_length].reshape(
            1, 1, candidate_length, candidate_length
        )
        candidate_allowed = candidate_allowed & candidate_attention_mask.to(dtype=torch.bool).reshape(
            batch, 1, 1, candidate_length
        )
        allowed = torch.cat((candidate_prefix_allowed, candidate_allowed), dim=-1)
        scores = torch.matmul(candidate_query, key.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = self.dropout(functional.softmax(scores, dim=-1))
        attended = torch.matmul(probabilities, value)
        candidate_attended = attended.transpose(1, 2).contiguous().reshape(batch, candidate_length, self.width)
        return self._project_pair(self.output, prefix_attended, candidate_attended)


class SwiGlu(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width = config["width"]
        feed_forward = config["feedForwardWidth"]
        self.gate = nn.Linear(width, feed_forward, bias=False)
        self.up = nn.Linear(width, feed_forward, bias=False)
        self.down = nn.Linear(feed_forward, width, bias=False)

        self.width = width

    def forward_pair(
        self,
        prefix: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_elements = prefix.shape[0] * prefix.shape[1]
        combined = torch.cat((
            prefix.reshape(-1, self.width),
            candidates.reshape(-1, self.width),
        ), dim=0)
        output = self.down(functional.silu(self.gate(combined)) * self.up(combined))
        return (
            output[:prefix_elements].reshape(prefix.shape),
            output[prefix_elements:].reshape(candidates.shape),
        )


class ContextTransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attention_norm = RmsNorm(config["width"], config["rmsNormEpsilon"])
        self.attention = RotarySelfAttention(config)
        self.feed_forward_norm = RmsNorm(config["width"], config["rmsNormEpsilon"])
        self.feed_forward = SwiGlu(config)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(
        self,
        prefix: torch.Tensor,
        candidates: torch.Tensor,
        prefix_mask: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_prefix = self.attention_norm(prefix)
        normalized_candidates = self.attention_norm(candidates)
        prefix_attention, candidate_attention = self.attention.forward_pair(
            normalized_prefix,
            normalized_candidates,
            prefix_mask,
            candidate_attention_mask,
            candidate_position_offset,
        )
        prefix = prefix + self.dropout(prefix_attention)
        candidates = candidates + self.dropout(candidate_attention)
        prefix_feed_forward, candidate_feed_forward = self.feed_forward.forward_pair(
            self.feed_forward_norm(prefix),
            self.feed_forward_norm(candidates),
        )
        prefix = prefix + self.dropout(prefix_feed_forward)
        candidates = candidates + self.dropout(candidate_feed_forward)
        return prefix, candidates


class ContextCandidateModel(nn.Module):
    """Scores only supplied candidates; it never exposes a free-form vocabulary head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.width = config["width"]
        self.candidate_length = config["maximumCandidateTokens"]
        self.prefix_length = config["sequenceLength"] - self.candidate_length
        self.token_embedding = nn.Embedding(config["vocabularySize"], self.width)
        self.field_embedding = nn.Embedding(config["fieldClasses"], self.width)
        self.dropout = nn.Dropout(config["dropout"])
        self.layers = nn.ModuleList(
            ContextTransformerBlock(config) for _ in range(config["layers"])
        )
        self.output_norm = RmsNorm(self.width, config["rmsNormEpsilon"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        field_class: torch.Tensor,
    ) -> torch.Tensor:
        prefix_ids = input_ids[:1, : self.prefix_length]
        candidate_ids = input_ids[:, self.prefix_length :]
        prefix_attention_mask = attention_mask[:1, : self.prefix_length]
        candidate_attention_mask = attention_mask[:, self.prefix_length :]
        prefix = self.token_embedding(prefix_ids)
        prefix = prefix + self.field_embedding(field_class[:1]).unsqueeze(1)
        candidates = self.token_embedding(candidate_ids)
        candidates = candidates + self.field_embedding(field_class).unsqueeze(1)
        prefix = self.dropout(prefix)
        candidates = self.dropout(candidates)
        for layer in self.layers:
            prefix, candidates = layer(
                prefix,
                candidates,
                prefix_attention_mask,
                candidate_attention_mask,
                self.prefix_length,
            )
        prefix = self.output_norm(prefix)
        candidates = self.output_norm(candidates)

        positions = torch.arange(self.prefix_length, device=input_ids.device).reshape(1, -1)
        last_prefix_position = prefix_attention_mask.sum(dim=-1, keepdim=True).clamp_min(1) - 1
        last_prefix = (
            prefix * (positions == last_prefix_position).to(prefix.dtype).unsqueeze(-1)
        ).sum(dim=1, keepdim=True)
        prediction_states = torch.cat((
            last_prefix.expand(candidates.shape[0], -1, -1),
            candidates[:, :-1, :],
        ), dim=1)
        target_embeddings = self.token_embedding(candidate_ids)
        token_scores = (prediction_states * target_embeddings).sum(dim=-1) / math.sqrt(self.width)
        scored_positions = candidate_mask[:, self.prefix_length :] * candidate_attention_mask.to(
            candidate_mask.dtype
        )
        score_sum = (token_scores * scored_positions).sum(dim=-1)
        return score_sum / scored_positions.sum(dim=-1).clamp_min(1.0)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
