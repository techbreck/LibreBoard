# SPDX-License-Identifier: GPL-3.0-only
"""Train-time definition of the layout-conditioned LibreBoard CTC encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SwipeCtcModel(nn.Module):
    """Scores live key slots instead of baking QWERTY letter identities into the weights."""

    def __init__(self, architecture: dict):
        super().__init__()
        self.path_points = architecture["pathPoints"]
        self.key_slots = architecture["keySlots"]
        self.output_frames = architecture["outputFrames"]
        self.width = architecture["width"]
        self.layout_temperature = architecture["layoutTemperature"]
        if self.path_points != self.output_frames * 2:
            raise ValueError("the v1 frame reducer requires two path points per output frame")

        self.path_projection = nn.Linear(5, self.width)
        self.path_position = nn.Parameter(torch.empty(self.path_points, self.width))
        self.key_encoder = nn.Sequential(
            nn.Linear(2, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=architecture["heads"],
            dim_feedforward=architecture["feedForwardWidth"],
            dropout=architecture["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=architecture["layers"])
        self.output_norm = nn.LayerNorm(self.width)
        self.blank_head = nn.Linear(self.width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.path_position, mean=0.0, std=0.02)

    def forward(self, path_coordinates: Tensor, key_centers: Tensor, key_mask: Tensor) -> Tensor:
        previous = torch.cat((path_coordinates[:, :1], path_coordinates[:, :-1]), dim=1)
        delta = path_coordinates - previous
        speed = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        path_features = torch.cat((path_coordinates, delta, speed), dim=-1)
        encoded_path = self.path_projection(path_features) + self.path_position.unsqueeze(0)

        encoded_keys = self.key_encoder(key_centers)
        squared_distance = torch.sum(
            (path_coordinates.unsqueeze(2) - key_centers.unsqueeze(1)) ** 2,
            dim=-1,
        )
        layout_scores = -self.layout_temperature * squared_distance
        layout_scores = layout_scores.masked_fill(key_mask.unsqueeze(1) <= 0, -10_000.0)
        layout_weights = torch.softmax(layout_scores, dim=-1)
        encoded_path = encoded_path + torch.matmul(layout_weights, encoded_keys)

        encoded_path = self.output_norm(self.encoder(encoded_path))
        frames = encoded_path.reshape(
            encoded_path.shape[0], self.output_frames, 2, self.width
        ).mean(dim=2)
        key_logits = torch.matmul(frames, encoded_keys.transpose(1, 2)) / math.sqrt(self.width)
        key_logits = key_logits.masked_fill(key_mask.unsqueeze(1) <= 0, -10_000.0)
        blank_logits = self.blank_head(frames)
        return torch.cat((blank_logits, key_logits), dim=-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
