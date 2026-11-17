#!/usr/bin/env python3
"""Prepare full Fashionpedia R50 configuration. No execution mode exists."""
import argparse
import json
from pathlib import Path

from fashion_grounding.segmentation.full_data_preparation import prepare, write_preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/runs/segmentation/full100_preparation_plan.json')
    parser.add_argument('--output', required=True, help='New local preparation directory')
    parser.add_argument('--manifest', help='Optional actual train image_manifest.json; JSON only')
    parser.add_argument('--annotations', help='Optional actual annotations_coco.json; JSON only')
    parser.add_argument('--illustrative-image-count', type=int,
                        help='Arithmetic example only; never a dataset receipt or launch config')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = prepare(root, args.plan, manifest=args.manifest, annotations=args.annotations,
                     illustrative_images=args.illustrative_image_count)
    write_preparation(args.output, report)
    print(json.dumps({'status': report['status'], 'ready_for_training': False,
                      'images_basis': report['data_identity']['basis'],
                      'max_updates': report['train_config_preparation']['max_updates'],
                      'output': str(Path(args.output).resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
