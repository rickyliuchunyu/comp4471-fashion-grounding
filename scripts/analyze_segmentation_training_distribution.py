"""Read accepted train annotations only; derive counts without images or models."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def area_band(area):
    if area <= 0:
        raise ValueError('Nonpositive mask area cannot enter the distribution')
    for limit, name in ((64, 'le64'), (256, 'gt64_le256'),
                        (1024, 'gt256_le1024'), (9216, 'gt1024_le9216')):
        if area <= limit:
            return name
    return 'gt9216'


def summarize(rows, images, total_instances):
    image_ids = {row['image_id'] for row in rows}
    groups = {images[i]['group_id'] for i in image_ids}
    areas = [row['area'] for row in rows]
    relative = [row['area'] / (images[row['image_id']]['width'] *
                               images[row['image_id']]['height']) for row in rows]
    bands = Counter(area_band(area) for area in areas)
    small = [row for row in rows if row['area'] <= 1024]
    tiny = [row for row in rows if row['area'] <= 256]
    return dict(instances=len(rows), images=len(image_ids), groups=len(groups),
        instance_share=len(rows) / total_instances if total_instances else None,
        image_share=len(image_ids) / len(images) if images else None,
        small_le1024=len(small), tiny_le256=len(tiny),
        small_images=len({row['image_id'] for row in small}),
        small_groups=len({images[row['image_id']]['group_id'] for row in small}),
        tiny_images=len({row['image_id'] for row in tiny}),
        small_fraction=len(small) / len(rows) if rows else None,
        relative_area_le1pct=sum(value <= .01 for value in relative),
        area_p10=quantile(areas, .1), area_p50=quantile(areas, .5),
        area_p90=quantile(areas, .9), relative_area_p50=quantile(relative, .5),
        **{key: bands[key] for key in ('le64', 'gt64_le256', 'gt256_le1024',
                                     'gt1024_le9216', 'gt9216')})


def analyze(coco, mapping):
    images = {image['id']: image for image in coco['images']}
    source = {row['source_id']: row for row in mapping['fashionpedia']
              if row['action'] == 'map'}
    rows = []
    excluded = Counter()
    for row in coco['annotations']:
        if row.get('iscrowd') or row.get('ignore'):
            excluded['crowd_or_ignore'] += 1
            continue
        if row['source_category_id'] not in source:
            raise ValueError('Unmapped source class in accepted annotations')
        expected = source[row['source_category_id']]
        if (expected['project_category_id'] != row['category_id'] or
                expected['source_name'] != row['source_category_name']):
            raise ValueError('Source-to-project grouping mismatch')
        if row['image_id'] not in images:
            raise ValueError('Annotation has no image dimensions/group')
        if any(images[row['image_id']][k] <= 0 for k in ('width', 'height')):
            raise ValueError('Invalid dimensions for relative area')
        rows.append(row)
    project = [dict(category_id=c['id'], category_name=c['name'],
        **summarize([r for r in rows if r['category_id'] == c['id']], images, len(rows)))
        for c in sorted(coco['categories'], key=lambda c: c['id'])]
    fine = [dict(source_category_id=sid, source_category_name=spec['source_name'],
        category_id=spec['project_category_id'],
        **summarize([r for r in rows if r['source_category_id'] == sid], images, len(rows)))
        for sid, spec in sorted(source.items())]
    return dict(total=summarize(rows, images, len(rows)),
        input_images=len(images), input_groups=len({i['group_id'] for i in images.values()}),
        input_annotations=len(coco['annotations']), excluded=dict(excluded),
        project_categories=project, source_categories=fine,
        definitions={
            'area': 'Previously verified original-image mask area; no mask decoding repeated.',
            'small': '0 < area <= 1024 original pixels; tiny <=256 is a nested subset.',
            'bands': 'Mutually exclusive upper-inclusive descriptive bins; not a new COCO AP computation.',
            'relative_area': 'mask area / original image area; <=1% is diagnostic, not COCO-small.',
            'images_groups': 'Distinct within each row; may overlap across class rows.',
            'zero_classes': 'All mapped source classes included; empty quantiles/fractions are null.',
            'limitations': 'Pre-augmentation counts, not visible instances after LSJ, visual quality or independent people/items.'})


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    package, output = Path(args.package), Path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory; never overwrite a result')
    hashes = {name: digest(package / name) for name in config['input_sha256']}
    if hashes != config['input_sha256']:
        raise ValueError('Input identity differs from the previously accepted package')
    coco = json.loads((package / 'annotations_coco.json').read_text())
    mapping = json.loads((package / 'mapping_config.json').read_text())
    result = analyze(coco, mapping)
    if (result['input_images'] != config['expected_images'] or
            result['total']['instances'] != config['expected_instances']):
        raise ValueError('Totals differ from reused acceptance evidence')
    previous = config['expected_project_counts']
    if {str(row['category_id']): row['instances'] for row in result['project_categories']} != previous:
        raise ValueError('Project totals differ from reused acceptance evidence')
    output.mkdir(parents=True)
    write_json(output / 'distribution.json', result)
    for name, rows in [('project_categories', result['project_categories']),
                       ('source_categories', result['source_categories'])]:
        with (output / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / 'resolved_config.json', config)
    elapsed = time.monotonic() - start
    if elapsed > config['budget_seconds']:
        raise TimeoutError('Analysis exceeded the registered budget')
    metadata = dict(run_id=config['run_id'], experiment_id='S0', owner='segmentation',
        executed_by='lcy', status='completed', split='train', candidate_source='none',
        git_commit=args.source_commit, script_sha256=digest(__file__), started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
        package=str(package), input_sha256=hashes, environment={'python':platform.python_version(),
        'dependencies':'Python standard library only'}, command=[sys.executable, *sys.argv],
        reused_evidence=config['reused_evidence'], model_loads=0, inference_images=0,
        optimizer_updates=0, images_opened=0, masks_decoded=0, old_smoke_runs=0,
        final_test_opened=False, language_queries_read=False,
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()})
    write_json(output / 'run_metadata.json', metadata)
    print(json.dumps({'run_id':config['run_id'], 'total':result['total'],
                      'elapsed_seconds':elapsed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
