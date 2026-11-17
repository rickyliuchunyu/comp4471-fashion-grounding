"""Explicit same-architecture model-only initialization, never training resume."""
from pathlib import Path

import torch

from ..data.schema import sha256
from .training_state import config_sha256, state_digest


def load_model_only(model, specification):
    path = Path(specification['path'])
    if specification['mode'] != 'model_only' or sha256(path) != specification['sha256']:
        raise ValueError('Model-only source checkpoint hash/mode mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    for field, expected in (
        ('completed_updates', specification['source_completed_updates']),
        ('data', specification['source_training_data']),
        ('config_sha256', specification['source_config_sha256']),
        ('git_commit', specification['source_git_commit']),
    ):
        if state[field] != expected:
            raise ValueError('Model-only source lineage mismatch: '+field)
    if config_sha256(state['config']) != state['config_sha256']:
        raise ValueError('Model-only source config is inconsistent')
    weights = state['model']
    if state_digest(weights) != state['component_digests']['model']:
        raise ValueError('Model-only source model fingerprint mismatch')
    live = model.state_dict()
    if live.keys() != weights.keys() or any(live[k].shape != weights[k].shape or live[k].dtype != weights[k].dtype for k in live):
        raise ValueError('Model-only initialization requires identical keys, shapes and dtypes')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Nonfinite model-only source')
    model.load_state_dict(weights, strict=True)
    return dict(mode='model_only', source=dict(specification), new_completed_updates=0,
                loaded_components=['model'],
                reset_components=['optimizer', 'scheduler', 'scaler', 'sampler', 'rng', 'amp_recovery'],
                model_digest=state_digest(model.state_dict()), classifier_reinitialized=False)
