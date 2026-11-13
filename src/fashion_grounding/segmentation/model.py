"""Pinned official Mask2Former construction and strict COCO head adaptation."""
from pathlib import Path
from collections import OrderedDict
import copy
import subprocess
import sys

import torch

UPSTREAM_COMMIT = '9b0651c6c1d5b3af2e6da0589b719c514ec0d69a'
CLASSIFIER_KEYS = {'sem_seg_head.predictor.class_embed.weight',
                   'sem_seg_head.predictor.class_embed.bias'}
LOSS_CLASS_WEIGHT_KEY = 'criterion.empty_weight'


def build_model(source, device='cuda'):
    source = Path(source).resolve()
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM_COMMIT:
        raise ValueError('Unverified Mask2Former commit')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Tracked upstream source modifications require review')
    sys.path.insert(0, str(source))
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model as d2_build
    from mask2former import add_maskformer2_config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(source / 'configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml'))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = ''  # Never implicitly download or load a legacy checkpoint.
    cfg.INPUT.FORMAT = 'RGB'
    cfg.freeze()
    model = d2_build(cfg)
    head = model.sem_seg_head.predictor.class_embed
    torch.nn.init.xavier_uniform_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return model, cfg


def adapt_coco_state(model, state):
    """Load COCO tensors, retaining the new classifier and class-count loss buffer.

    File deserialization/provenance remains the caller's responsibility. Course
    checkpoints use strict model.load_state_dict instead of this initialization.
    """
    current = model.state_dict()
    state = OrderedDict(state)
    # Exact upstream v1 -> v2 migration; never use a fuzzy key matcher.
    old_query = 'sem_seg_head.predictor.static_query.weight'
    new_query = 'sem_seg_head.predictor.query_feat.weight'
    renamed = {}
    if old_query in state:
        if new_query in state:
            raise ValueError('Ambiguous legacy/current query feature keys')
        state[new_query] = state.pop(old_query)
        renamed[old_query] = new_query
    if set(state) != set(current):
        raise ValueError(f'State keys differ: missing={sorted(set(current)-set(state))}, '
                         f'unexpected={sorted(set(state)-set(current))}')
    mismatch = {k for k in current if tuple(state[k].shape) != tuple(current[k].shape)}
    skipped = set(CLASSIFIER_KEYS)
    if LOSS_CLASS_WEIGHT_KEY in current:
        old, new = torch.as_tensor(state[LOSS_CLASS_WEIGHT_KEY]), current[LOSS_CLASS_WEIGHT_KEY].cpu()
        if tuple(old.shape) != (81,) or tuple(new.shape) != (9,):
            raise ValueError('Expected 81 -> 9 loss class weights')
        if not torch.equal(old[:-1], torch.ones_like(old[:-1])) or not torch.equal(new[:-1], torch.ones_like(new[:-1])) or old[-1] != new[-1]:
            raise ValueError('Unexpected COCO no-object loss weight')
        skipped.add(LOSS_CLASS_WEIGHT_KEY)
    if mismatch != skipped:
        raise ValueError(f'Expected only class-count dependent mismatch, got {sorted(mismatch)}')
    for key in CLASSIFIER_KEYS:
        if state[key].shape[0] != 81 or current[key].shape[0] != 9 or state[key].shape[1:] != current[key].shape[1:]:
            raise ValueError('Expected COCO 80+1 -> project 8+1 head')
    for key in current:
        tensor = torch.as_tensor(state[key])
        if tensor.dtype != current[key].dtype or not torch.isfinite(tensor).all():
            raise ValueError(f'Unexpected dtype/non-finite checkpoint tensor: {key}')
    retained = {k: current[k].detach().cpu().clone() for k in skipped}
    weights = OrderedDict((k, torch.as_tensor(v)) for k, v in state.items() if k not in skipped)
    # Use the constructed model's versions after the explicit legacy conversion.
    weights._metadata = copy.deepcopy(current._metadata)
    incompatible = model.load_state_dict(weights, strict=False)
    if set(incompatible.missing_keys) != skipped or incompatible.unexpected_keys:
        raise ValueError('Unexpected checkpoint load mismatch')
    loaded = model.state_dict()
    for key, expected in {**weights, **retained}.items():
        if not torch.equal(loaded[key].detach().cpu(), expected):
            raise ValueError(f'Post-load tensor equality failed: {key}')
    return dict(skipped_classifier_keys=sorted(CLASSIFIER_KEYS),
                retained_loss_buffers=sorted(skipped - CLASSIFIER_KEYS),
                renamed_keys=renamed, unexpected_keys=[],
                loaded_tensor_count=len(weights), exact_tensor_equality_verified=True,
                retained_tensors_unchanged=True,
                shapes={k: dict(source=list(state[k].shape), target=list(current[k].shape))
                        for k in sorted(skipped)})


def check_cuda_extension():
    """Check actual CUDA forward AND gradients against upstream PyTorch reference.

    An import alone cannot detect a binary compiled for the wrong GPU architecture.
    Requires build_model() to have registered the pinned upstream source first.
    """
    from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction, ms_deform_attn_core_pytorch)
    shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long, device='cuda')
    starts = torch.tensor([0, 4], dtype=torch.long, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(7)
    value = torch.rand(1, 5, 2, 2, generator=generator, device='cuda', dtype=torch.double, requires_grad=True)
    locations = torch.rand(1, 2, 2, 2, 2, 2, generator=generator, device='cuda', dtype=torch.double, requires_grad=True)
    weights = torch.rand(1, 2, 2, 2, 2, generator=generator, device='cuda', dtype=torch.double, requires_grad=True)
    reference = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 1)
    torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-6)
    expected_grad = torch.autograd.grad(reference, (value, locations, weights), grad_outputs=torch.ones_like(reference).contiguous())
    actual_grad = torch.autograd.grad(actual, (value, locations, weights), grad_outputs=torch.ones_like(actual).contiguous())
    for observed, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, expected, atol=1e-8, rtol=1e-6)
    return dict(forward_max_abs_error=float((actual-reference).abs().max()),
                gradient_max_abs_errors=[float((a-b).abs().max()) for a,b in zip(actual_grad, expected_grad)],
                gpu_capability=list(torch.cuda.get_device_capability()), status='passed')
