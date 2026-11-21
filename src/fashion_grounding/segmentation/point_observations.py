"""Observe the pinned Mask2Former final mask loss without drawing extra points."""
from contextlib import contextmanager
from importlib import import_module
from unittest.mock import patch

import numpy as np


def point_hits(mask, coords, labels):
    """Foreground uses actual bilinear GT labels. Boundary uses the containing pixel.

    The boundary band spans the inner and outer one-pixel 8-neighbour contour,
    with background outside the image. Foreground and boundary may overlap.
    """
    h, w = mask.shape
    padded = np.pad(mask, 1)
    eroded, dilated = mask.copy(), mask.copy()
    for dy in range(3):
        for dx in range(3):
            neighbour = padded[dy:dy+h, dx:dx+w]
            eroded &= neighbour
            dilated |= neighbour
    x = np.clip((coords[:, 0] * w).astype(int), 0, w-1)
    y = np.clip((coords[:, 1] * h).astype(int), 0, h-1)
    return dict(points=len(coords), foreground_hits=int((labels >= .5).sum()),
                boundary_hits=int((dilated[y, x] & ~eroded[y, x]).sum()),
                augmented_area=int(mask.sum()))


@contextmanager
def observe_final_points(criterion, batch, rows):
    """Scope observation to the first (final decoder layer) loss_masks call.

    This single-process trainer evaluates the final masks before any auxiliary
    losses. Only the GT point_sample call is inspected; both original calls
    return their original tensors. Contexts restore both bindings on failure.
    """
    original_loss = criterion.loss_masks
    module = import_module(original_loss.__module__)
    original_sample = module.point_sample
    observed = False

    def loss(outputs, targets, indices, num_masks):
        nonlocal observed
        if observed:
            return original_loss(outputs, targets, indices, num_masks)
        observed = True
        owners = [(i, j) for i, (_, gt) in enumerate(indices) for j in gt.tolist()]
        sampled = False

        def sample(tensor, coords, **kw):
            nonlocal sampled
            values = original_sample(tensor, coords, **kw)
            if not sampled:
                sampled = True
                if len(owners) != len(coords):
                    raise ValueError('Point observation target ordering differs')
                coordinates = coords.detach().float().cpu().numpy()
                labels = values[:, 0].detach().float().cpu().numpy()
                for k, (i, j) in enumerate(owners):
                    item = batch[i]
                    mask = tensor[k, 0].detach().bool().cpu().numpy()
                    rows.append(dict(image_id=item['image_id'], **item['point_targets'][j],
                                     **point_hits(mask, coordinates[k], labels[k])))
            return values

        with patch.object(module, 'point_sample', sample):
            result = original_loss(outputs, targets, indices, num_masks)
        if not sampled:
            raise RuntimeError('Expected the upstream mask loss to sample target points')
        return result

    with patch.object(criterion, 'loss_masks', loss):
        yield
    if not observed:
        raise RuntimeError('Final mask loss was not observed')
