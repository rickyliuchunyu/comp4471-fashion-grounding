"""Training configuration, optimizer, schedule, sampling and resumable state."""
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


def read_config(path):
    path = Path(path)
    config = json.loads(path.read_text())
    if 'base_config' in config:
        base = json.loads((path.parent / config.pop('base_config')).read_text())
        if 'base_config' in base:
            raise ValueError('Only one configuration inheritance level is supported')
        config = {**base, **config}
    for key in ('max_updates', 'microbatch_images', 'accumulation_steps', 'effective_batch_images',
                'image_size', 'checkpoint_interval', 'budget_seconds'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'Expected positive integer {key}')
    if config['microbatch_images'] * config['accumulation_steps'] != config['effective_batch_images']:
        raise ValueError('Effective batch differs from microbatch x accumulation')
    if config['image_size'] % 32 or config['warmup_updates'] < 0:
        raise ValueError('Invalid spatial size/warmup')
    if config['lr_milestones'] != sorted(set(config['lr_milestones'])):
        raise ValueError('Learning-rate milestones must be unique and increasing')
    for key in ('image_manifest_sha256', 'annotation_sha256'):
        value = config[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Pin the real training data before running: {key}')
    for key in ('learning_rate', 'backbone_lr_multiplier', 'gradient_clip_norm', 'amp_initial_scale'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 0 < config['warmup_start_factor'] <= 1 or not 0 < config['lr_decay'] <= 1:
        raise ValueError('Invalid LR multiplier')
    if not 0 < config['min_scale'] <= config['max_scale'] or config['weight_decay'] < 0:
        raise ValueError('Invalid augmentation/decay')
    if config['purpose'] == 'pilot_200_update_stability' and config['max_updates'] != 200:
        raise ValueError('The pilot stability profile must stop at 200 updates')
    if config['purpose'] not in ('pilot_200_update_stability', 's0_training'):
        raise ValueError('Unknown training purpose')
    return config


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_optimizer(model, config):
    """Upstream Mask2Former parameter-group policy; clipping happens after unscale."""
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                  nn.LayerNorm, nn.LocalResponseNorm)
    groups, seen = [], set()
    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or param in seen:
                continue
            seen.add(param)
            lr = config['learning_rate'] * (config['backbone_lr_multiplier'] if 'backbone' in module_name else 1)
            decay = config['weight_decay']
            if 'relative_position_bias_table' in name or 'absolute_pos_embed' in name:
                decay = 0.0
            if isinstance(module, norm_types):
                decay = config['weight_decay_norm']
            if isinstance(module, nn.Embedding):
                decay = config['weight_decay_embed']
            groups.append(dict(params=[param], lr=lr, weight_decay=decay,
                               parameter_name=f'{module_name}.{name}'))
    return torch.optim.AdamW(groups, lr=config['learning_rate'])


class UpdateSchedule:
    """completed_updates=0 sets the LR for update 1; only successful steps advance."""
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.warmup = config['warmup_updates']
        self.start = config['warmup_start_factor']
        self.milestones = list(config['lr_milestones'])
        self.gamma = config['lr_decay']
        self.completed_updates = 0
        self._apply()

    def _apply(self):
        alpha = min(self.completed_updates / self.warmup, 1.) if self.warmup else 1.
        factor = (self.start * (1-alpha) + alpha) * self.gamma ** bisect_right(self.milestones, self.completed_updates)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * factor

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return {key: getattr(self, key) for key in ('base_lrs', 'warmup', 'start', 'milestones', 'gamma', 'completed_updates')}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state[k] != expected[k] for k in expected if k != 'completed_updates'):
            raise ValueError('Resume scheduler configuration mismatch')
        self.completed_updates = state['completed_updates']
        self._apply()


class StatefulSampler:
    """Single-process shuffled epochs with no worker prefetch to lose on resume."""
    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError('Empty training dataset')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.position, self.epoch = 0, 0

    def next(self):
        if self.position == self.size:
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.position, self.epoch = 0, self.epoch + 1
        index = self.order[self.position]
        self.position += 1
        return index

    def state_dict(self):
        return dict(size=self.size, order=list(self.order), position=self.position, epoch=self.epoch,
                    generator=self.generator.get_state())

    def load_state_dict(self, state):
        if state['size'] != self.size or sorted(state['order']) != list(range(self.size)) or not 0 <= state['position'] <= self.size:
            raise ValueError('Invalid sampler resume state')
        self.order, self.position, self.epoch = list(state['order']), state['position'], state['epoch']
        self.generator.set_state(state['generator'])


def capture_rng():
    name, keys, pos, gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(name, torch.tensor(keys.astype(np.int64)), pos, gauss, cached),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, pos, gauss, cached = state['numpy']
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, gauss, cached))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def state_digest(value):
    """Device-independent recursive fingerprint for exact state restoration checks."""
    digest = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            tensor = v.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=repr):
                digest.update(repr(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(type(v).__name__.encode())
            for item in v: visit(item)
        else:
            digest.update(repr(v).encode())
    visit(value)
    return digest.hexdigest()


def save_checkpoint(path, state):
    """Publish a new checkpoint and hash sidecar without overwriting old checkpoints."""
    path = Path(path)
    partial = path.with_name(path.name + f'.{os.getpid()}.partial')
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError(path)
    try:
        with partial.open('xb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    from ..data.schema import sha256
    digest = sha256(path)
    with path.with_suffix('.sha256').open('x') as stream:
        stream.write(digest+'\n')
    return digest


def load_checkpoint(path, expected_config_sha256, expected_data):
    from ..data.schema import sha256
    path = Path(path)
    if sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Resume checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['config_sha256'] != expected_config_sha256 or state['data'] != expected_data:
        raise ValueError('Resume configuration or dataset mismatch')
    if state['completed_updates'] != state['scheduler']['completed_updates']:
        raise ValueError('Checkpoint update/schedule mismatch')
    return state
