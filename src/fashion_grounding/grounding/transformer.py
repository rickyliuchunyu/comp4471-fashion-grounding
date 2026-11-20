"""Load and score the attribute-supervised candidate Transformer from a frozen release."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fashion_grounding.data.schema import read_json, require, sha256
from .heads import AppearanceResidual


class AttributeTransformer(nn.Module):
    """One query token and one token per candidate, with mask color and spatial cues."""

    def __init__(self, dimension):
        super().__init__()
        self.appearance = AppearanceResidual(dimension)
        self.query_projection = nn.Linear(dimension, 64)
        self.region_projection = nn.Linear(dimension + 24, 64)
        self.encoder = nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.readout = nn.Linear(64, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.query_norm = nn.LayerNorm(64)
        self.region_norm = nn.LayerNorm(64)
        self.modulation = nn.Linear(64, 64)
        self.register_buffer("cue_map", torch.zeros(dimension + 1, 13))
        self.gains = nn.Parameter(torch.tensor([4.0, 4.0]))

    def forward(self, visual, text, geometry):
        query = self.query_norm(self.query_projection(text)).unsqueeze(1)
        regions = self.region_norm(self.region_projection(torch.cat([visual, geometry], dim=-1)))
        regions = regions * (1 + self.modulation(query).tanh())
        padding = visual.square().sum(-1) == 0
        padding = torch.cat([torch.zeros_like(padding[:, :1]), padding], dim=1)
        tokens = self.encoder(torch.cat([query, regions], dim=1), src_key_padding_mask=padding)

        cues = (torch.cat([text, torch.ones_like(text[:, :1])], dim=-1) @ self.cue_map).clamp(-1, 1)
        horizontal = (2 * geometry[:, :, 10] - 1) * cues[:, :1]
        color = (geometry[:, :, 12:] * cues[:, None, 1:]).sum(-1)
        appearance = self.appearance(visual, text)
        context = self.readout(tokens[:, 1:]).squeeze(-1)
        return appearance + context + self.gains[0] * horizontal + self.gains[1] * color


class TransformerRanker:
    def __init__(self, model, category_map, weights):
        self.model = model
        self.category_map = category_map
        self.weights = weights
        self.attribute_map = model.cue_map.detach().numpy()

    @torch.inference_mode()
    def score(self, visual, text, geometry, base, candidates):
        """Score in manifest order using only the query and candidate features."""
        residual = self.model(
            torch.from_numpy(visual)[None], torch.from_numpy(text), torch.from_numpy(geometry)[None]
        ).numpy()[0]
        text_with_bias = np.append(text.reshape(-1), 1.0).astype(np.float32)
        category_indices = np.asarray([c.category_id for c in candidates], dtype=int) - 1
        category = (text_with_bias @ self.category_map).clip(0, 1)[category_indices]
        direction = (text_with_bias @ self.attribute_map)[0].clip(-1, 1)
        probability = np.asarray([c.score if c.score is not None else 0.5 for c in candidates],
                                 dtype=np.float32).clip(0.05, 0.95)
        confidence = np.log(probability / (1 - probability))
        position = category * direction * (2 * geometry[:, 4] - 1)
        calibration = np.stack((category, confidence, position), axis=-1)
        return (base + residual + calibration @ self.weights).astype(np.float32)


def load_transformer(release, signature, method, seed=None):
    """Verify the model/encoder pair before consuming any queries."""
    require(method in release["calibration"], "unknown Transformer method")
    seed = str(release["default_seed"] if seed is None else seed)
    require(seed in release["checkpoints"], "seed is not in the Transformer release")

    encoder = release["encoder"]
    encoder_path = Path(encoder["path"])
    require(sha256(encoder_path) == encoder["sha256"], "Transformer encoder signature hash differs")
    require(signature == read_json(encoder_path), "Transformer requires the released encoder signature")

    checkpoint = release["checkpoints"][seed]
    checkpoint_path = Path(checkpoint["path"])
    require(sha256(checkpoint_path) == checkpoint["sha256"], "Transformer checkpoint hash differs")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dimension = signature["dimension"]
    require(state["dimension"] == dimension, "incompatible Transformer embedding dimension")
    model = AttributeTransformer(dimension)
    model.load_state_dict(state["model"], strict=True)
    model.eval().requires_grad_(False)

    classifier = release["category_map"]
    classifier_path = Path(classifier["path"])
    require(sha256(classifier_path) == classifier["sha256"], "Transformer category map hash differs")
    category_map = np.load(classifier_path, allow_pickle=False)
    require(category_map.dtype == np.float32 and category_map.shape == (dimension + 1, 8)
            and np.isfinite(category_map).all(), "invalid Transformer category map")
    weights = np.asarray(release["calibration"][method])
    require(weights.shape == (3,) and np.isfinite(weights).all(), "invalid Transformer calibration")
    metadata = {
        "seed": int(seed),
        "checkpoint_sha256": checkpoint["sha256"],
        "encoder_signature_sha256": encoder["sha256"],
        "category_map_sha256": classifier["sha256"],
        "calibration_weights": weights.tolist(),
    }
    return TransformerRanker(model, category_map, weights), metadata
