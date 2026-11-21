"""Read-only segmentation acceptance of a pinned published train pilot."""
import argparse
from pathlib import Path
import subprocess

from fashion_grounding.data.schema import read_json, sha256, safe_path, require, validate_package, write_json
from fashion_grounding.segmentation.data import load_training_package, map_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'config', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    package, output = Path(args.package), Path(args.output)
    config = read_json(args.config)
    require(config['optimizer_updates'] == 2, 'Acceptance GPU profile must remain two updates')
    require(not output.exists(), 'Use a new acceptance output directory')
    pinned = config['package_sha256']
    for name, expected in pinned.items():
        require(sha256(safe_path(package, name)) == expected, 'Pinned input changed: '+name)
    checksums = read_json(package/'SHA256SUMS.json')
    require(bool(checksums), 'Empty published checksum list')
    for name, expected in checksums.items():
        require(sha256(safe_path(package, name)) == expected, 'Published file checksum mismatch: '+name)
    schema = validate_package(package)
    records, loaded = load_training_package(package)
    mapped = [map_record(r, config['short_edge'], config['max_size']) for r in records]
    rows = []
    for original, resized in zip(records, mapped):
        require(resized['image'].shape[0] == 3, 'Expected RGB channels')
        instances = resized['instances']
        require(instances.gt_masks.shape[0] == len(instances.gt_classes), 'Mask/class alignment')
        require(all(0 <= int(c) <= 7 for c in instances.gt_classes), 'Invalid internal class ID')
        rows.append(dict(image_id=original['image_id'], original_hw=[original['height'], original['width']],
            mapped_hw=list(resized['image'].shape[1:]), original_instances=len(original['annotations']),
            mapped_instances=len(instances)))
    require(all(sha256(safe_path(package,n)) == h for n,h in checksums.items()), 'Source changed during acceptance')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'package_acceptance.json', dict(status='passed', source_package=package.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha256(args.config), checksum_manifest_sha256=sha256(package/'SHA256SUMS.json'),
        pinned_hashes=pinned, published_file_hashes_verified=len(checksums), schema=schema, loader=loaded,
        mapped_images=len(rows), mapped_instances=sum(r['mapped_instances'] for r in rows), image_checks=rows,
        source_hashes_unchanged=True, gpu_used=False,
        scope='Segmentation data consumption; human annotation quality and dev freeze are not adjudicated'))
    print(dict(status='passed', checksum_files=len(checksums), images=len(rows),
               instances=loaded['instances'], mapped_instances=sum(r['mapped_instances'] for r in rows)))


if __name__ == '__main__':
    main()
