"""Additional visual evidence and a learned text-to-candidate interaction."""
import numpy as np
import torch
from torch import nn

from .geometry import geometry_features
from .heads import AppearanceResidual


def binding_geometry(image, candidates):
    """Append a normalized mask-only histogram to the original geometry."""
    geometry = geometry_features(candidates, image.size)
    hsv = np.asarray(image.convert("HSV"))
    hue = hsv[:, :, 0].astype(np.int16) * 8 // 256
    neutral = (hsv[:, :, 2] < 42) | (hsv[:, :, 1] < 46)
    brightness = hsv[:, :, 2] // 64
    indices = np.where(neutral, 8 + brightness, hue)
    histogram = np.zeros((len(candidates), 12), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        pixels = indices[candidate.mask]
        histogram[i] = np.bincount(pixels, minlength=12) / len(pixels)
    return np.concatenate((geometry, histogram), axis=1)


class BindingResidual(nn.Module):
    """A learned query weights geometry and candidate-mask color features."""

    def __init__(self, dimension):
        super().__init__()
        self.appearance = AppearanceResidual(dimension)
        self.binding = nn.Linear(dimension, 24)
        nn.init.zeros_(self.binding.weight)
        nn.init.zeros_(self.binding.bias)

    def spatial_scores(self, text, geometry):
        return 8 * (self.binding(text)[:, None, :] * geometry).sum(-1)

    def forward(self, visual, text, geometry):
        return self.appearance(visual, text) + self.spatial_scores(text, geometry)
