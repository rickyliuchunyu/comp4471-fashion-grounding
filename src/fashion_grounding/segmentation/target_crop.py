"""Train-only, context-preserving crops. Every annotation uses the same transform."""
import random
import numpy as np
from pycocotools import mask as mask_api


def validate_crop(policy):
    if (policy.get('version') != 1 or not 0 <= policy.get('probability', -1) <= .25
            or policy.get('max_source_area') != 1024
            or not .5 <= policy.get('min_context_fraction', 0) <= 1
            or policy.get('bbox_context_multiplier', 0) < 3):
        raise ValueError('Invalid bounded target-preserving crop policy')


def target_transform(image, annotations, config):
    """Return None for the original LSJ path, without consuming its NumPy RNG."""
    policy = config.get('target_crop')
    if policy is None:
        return None
    validate_crop(policy)
    if random.random() >= policy['probability']:
        return None
    eligible = [a for a in annotations if
        0 < a.get('source_area', a.get('area', float('inf'))) <= policy['max_source_area']]
    if not eligible:
        return None
    from detectron2.data import transforms as T
    target = eligible[int(np.random.randint(len(eligible)))]
    # Use the actual source mask extent; never infer a smaller box from its area.
    mask = mask_api.decode(target['segmentation'])
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Empty target in accepted input')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    side = int(np.ceil(max(max(h,w)*policy['min_context_fraction'],
                           max(x2-x1,y2-y1)*policy['bbox_context_multiplier'])))
    cw, ch = min(w,side), min(h,side)
    # Jitter inside the feasible interval while retaining the complete target bbox.
    x = int(np.random.randint(max(0,x2-cw), min(x1,w-cw)+1))
    y = int(np.random.randint(max(0,y2-ch), min(y1,h-ch)+1))
    size = config['image_size']
    scale = size/max(ch,cw)
    nh, nw = int(ch*scale+.5), int(cw*scale+.5)
    transforms = [T.CropTransform(x,y,cw,ch), T.ResizeTransform(ch,cw,nh,nw)]
    if config['random_flip'] == 'horizontal' and np.random.rand() < .5:
        transforms.append(T.HFlipTransform(nw))
    elif config['random_flip'] not in ('horizontal','none'):
        raise ValueError('Unsupported flip policy')
    transforms.append(T.PadTransform(0,0,size-nw,size-nh,pad_value=128,seg_pad_value=0))
    transforms = T.TransformList(transforms)
    transforms.target_annotation_id = target.get('annotation_id',target.get('id'))
    transforms.view_kind = 'target_context'
    return transforms.apply_image(image), transforms
