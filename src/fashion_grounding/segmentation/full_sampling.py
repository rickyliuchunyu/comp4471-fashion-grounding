"""Fixed one-pass image order with interleaved, group-capped small-instance replay."""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import random


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected):
    data = Path(path).read_bytes()
    if digest(data) != expected:
        raise ValueError(f'Input fingerprint differs: {path}')
    return json.loads(data)


def target_slot(slot, total, targeted):
    """Evenly spread targeted draws across all slots, including the last one."""
    return (slot + 1) * targeted // total != slot * targeted // total


def build_order(images, annotations, protection, config):
    policy = config['sampling']
    names = policy['source_names']
    target_count = policy['target_draws']
    if not names or len(set(names)) != len(names) or target_count % len(names):
        raise ValueError('Target draws must split evenly across distinct source classes')
    groups = {im['image_id']: im['group_id'] for im in images}
    if len(groups) != len(images) or len(images) != config['training_images']:
        raise ValueError('Training membership differs')
    if (len(protection['images']) != protection['distinct_images'] or
            not {'train', 'test'} <= set(protection['forbidden_splits'])):
        raise ValueError('Invalid exposure protection list')
    blocked = {field: {row[field] for row in protection['images']} for field in ('image_id', 'sha256', 'group_id', 'source_key')}
    for im in images:
        if (im['split'] != 'train' or im['source_split'] != 'official_train2020'
            or 'train-' + str(im['source_image_id']) in blocked['source_key']
            or any(im[k] in blocked[k] for k in ('image_id', 'sha256', 'group_id'))):
            raise ValueError('Protected or nontraining image in sampling population')

    pools = {name: defaultdict(set) for name in names}
    for ann in annotations:
        name = ann['source_category_name']
        if name in pools and not ann['iscrowd'] and 0 < ann['area'] <= policy['max_area']:
            image = ann['image_id']
            pools[name][groups[image]].add(image)
    rng = random.Random(config['seed'])
    uniform = sorted(groups)
    rng.shuffle(uniform)
    # Lazy heaps prioritize the least-replayed group across all pools. Ties are
    # seeded, and multiple eligible images within a group share its single cap.
    queues = {}
    for name in names:
        queues[name] = [(0, rng.random(), group) for group in sorted(pools[name])]
        heapq.heapify(queues[name])
    uses, image_uses = Counter(), Counter()
    targeted = []
    for draw in range(target_count):
        name = names[draw % len(names)]
        queue = queues[name]
        while queue:
            last, _, group = heapq.heappop(queue)
            if uses[group] >= policy['max_extra_per_group']:
                continue
            if last == uses[group]:
                break
            heapq.heappush(queue, (uses[group], rng.random(), group))
        else:
            raise ValueError('Insufficient eligible groups under replay cap: ' + name)
        choices = sorted(pools[name][group])
        minimum = min(image_uses[i] for i in choices)
        selected = rng.choice([i for i in choices if image_uses[i] == minimum])
        targeted.append(selected)
        image_uses[selected] += 1
        uses[group] += 1
        heapq.heappush(queue, (uses[group], rng.random(), group))
    total = len(uniform) + len(targeted)
    regular, extra = iter(uniform), iter(targeted)
    order = [next(extra if target_slot(i, total, target_count) else regular) for i in range(total)]
    report = dict(uniform_draws=len(uniform), targeted_draws=len(targeted), distinct_target_images=len(image_uses),
                  distinct_target_groups=len(uses), max_extra_per_image=max(image_uses.values()),
                  max_extra_per_group=max(uses.values()), pools={})
    for name in names:
        report['pools'][name] = dict(images=sum(map(len, pools[name].values())), groups=len(pools[name]),
                                     targeted_draws=target_count // len(names))
    return order, report


class FixedOrderSampler:
    """Positions in the immutable schedule are also the AMP retry/resume cursor."""
    epoch = 0

    def __init__(self, record_ids, config):
        spec = config['sampling']
        saved = checked_json(spec['order_path'], spec['order_sha256'])
        if (saved['train_manifest_sha256'] != config['image_manifest_sha256']
                or saved['train_annotation_sha256'] != config['annotation_sha256']
                or saved['protection_sha256'] != spec['protection_sha256'] or saved['seed'] != config['seed']):
            raise ValueError('Sampling order provenance differs')
        indices = {image_id: i for i, image_id in enumerate(record_ids)}
        order = saved['image_ids']
        if (len(indices) != len(record_ids) or set(order) != indices.keys()
                or len(order) != config['max_updates'] * config['effective_batch_images']):
            raise ValueError('Sampling order membership or length differs')
        self.indices = [indices[i] for i in order]
        self.sha = spec['order_sha256']
        self.position = 0

    def next(self):
        if self.position == len(self.indices):
            raise ValueError('Sampling schedule exhausted')
        index = self.indices[self.position]
        self.position += 1
        return index

    def state_dict(self):
        return dict(order_sha256=self.sha, position=self.position)

    def load_state_dict(self, state):
        p = state['position']
        if state['order_sha256'] != self.sha or type(p) is not int or not 0 <= p <= len(self.indices):
            raise ValueError('Wrong order or cursor')
        self.position = p
