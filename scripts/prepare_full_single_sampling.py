"""Build and verify the new full-release sampler using annotation metadata only."""
import argparse
from collections import Counter
from pathlib import Path
import json

from fashion_grounding.segmentation.full_sampling import build_order, checked_json, digest, FixedOrderSampler, target_slot


def prepare(package, protection, config, output):
    spec = config['sampling']
    images = checked_json(package/'image_manifest.json', config['image_manifest_sha256'])['images']
    annotations = checked_json(package/'annotations_coco.json', config['annotation_sha256'])['annotations']
    registry = checked_json(protection, spec['protection_sha256'])
    order, receipt = build_order(images, annotations, registry, config)
    batch = config['effective_batch_images']
    updates = config['max_updates']
    names = spec['source_names']
    cycle = spec.get('target_cycle', names)
    total = len(order)
    extra = spec['target_draws']
    if total != updates * batch:
        raise ValueError('Exposure budget differs')
    # Validate the finished stream against COCO IDs independently of group queues.
    eligible = {name: set() for name in names}
    groups = {im['image_id']: im['group_id'] for im in images}
    for a in annotations:
        if a['source_category_name'] in eligible and not a['iscrowd'] and 0 < a['area'] <= spec['max_area']:
            eligible[a['source_category_name']].add(a['image_id'])
    extras, regular, half = [], [], {}
    for i, image in enumerate(order):
        if target_slot(i, total, extra):
            if image not in eligible[cycle[len(extras) % len(cycle)]]:
                raise ValueError('Wrong targeted draw')
            extras.append(image)
        else:
            regular.append(image)
        if i + 1 == total // 2:
            half = dict(uniform=len(regular), targeted=len(extras))
    if (Counter(regular) != Counter(groups.keys()) or len(extras) != extra
            or max(Counter(groups[i] for i in extras).values()) > spec['max_extra_per_group']):
        raise ValueError('Coverage, quota or repeat cap differs')
    saved = dict(train_manifest_sha256=config['image_manifest_sha256'], train_annotation_sha256=config['annotation_sha256'],
                 protection_sha256=spec['protection_sha256'], seed=config['seed'], image_ids=order)
    data = (json.dumps(saved, separators=(',', ':')) + '\n').encode()
    order_sha = digest(data)
    if spec.get('order_sha256', order_sha) != order_sha:
        raise ValueError('Rebuilt schedule differs from prepared version')
    output.mkdir(parents=True, exist_ok=True)
    file = output/'order.json'
    with file.open('xb') as stream:
        stream.write(data)
    local_config = dict(config, sampling=dict(spec, order_path=str(file), order_sha256=order_sha))
    ids = sorted(groups)
    sampler = FixedOrderSampler(ids, local_config)
    def take(n):
        return [ids[sampler.next()] for _ in range(n)]
    if take(total//2) != order[:total//2]:
        raise ValueError("Sampler replay differs from scheduled image IDs")
    midpoint = sampler.state_dict()
    retry = take(batch)
    sampler.load_state_dict(midpoint)
    if take(batch) != retry:
        raise ValueError("Sampler replay differs from scheduled image IDs")
    sampler = FixedOrderSampler(ids, local_config)
    sampler.load_state_dict(midpoint)
    if take(total//2) != order[total//2:]:
        raise ValueError("Sampler replay differs from scheduled image IDs")
    try:
        sampler.next()
    except ValueError:
        pass
    else:
        raise ValueError('Schedule continued beyond budget')
    receipt.update(order_sha256=order_sha, total_draws=total, distinct_images=len(set(order)),
                   half_update=updates//2, half_draws=half, protection_sha256=spec['protection_sha256'],
                   replayed_batch_equal=True, resumed_suffix_equal=True, rejects_overrun=True)
    (output/'sampling_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'protection', 'config', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.package, a.protection, json.loads(a.config.read_text()), a.output), indent=2))
