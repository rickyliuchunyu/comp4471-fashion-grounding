"""New schedule, provenance and preparation-only boundaries; no model smoke."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from fashion_grounding.segmentation.full_data_preparation import (
    epoch_schedule, file_hash, metadata_identity, prepare, read_json, write_preparation,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/runs/segmentation/full100_preparation_plan.json'
RECIPE = read_json(ROOT / 'configs/runs/segmentation/full100_train_recipe.json')
CATEGORIES = read_json(ROOT / 'configs/categories.json')['categories']


class FullDataPreparationTests(unittest.TestCase):
    def test_cumulative_rounding_and_decay_boundaries(self):
        schedule = epoch_schedule(40001, RECIPE)
        self.assertEqual(schedule['max_updates'], 250007)
        self.assertEqual(schedule['lr_milestones'], [200005, 225006])
        self.assertEqual(schedule['rounding_extra_images'], 12)
        self.assertEqual(len(schedule['checkpoint_updates']), 100)
        self.assertEqual(len(schedule['evaluation_nodes']), 50)
        self.assertEqual(schedule['evaluation_nodes'][-1], {'epoch': 100, 'update': 250007})
        self.assertEqual(schedule['checkpoint_updates'][79], 200005)
        # ceil(N/B)*100 would silently introduce 1,500 unnecessary image exposures.
        self.assertNotEqual(schedule['max_updates'], ((40001 + 15) // 16) * 100)

    def test_unknown_full_population_stays_unresolved_and_non_executable(self):
        report = prepare(ROOT, PLAN)
        self.assertIsNone(report['schedule'])
        self.assertIsNone(report['train_config_preparation']['max_updates'])
        self.assertIsNone(report['train_config_preparation']['budget_seconds'])
        self.assertFalse(report['ready_for_training'])
        self.assertFalse(report['training_authorized'])
        self.assertNotIn('torch', sys.modules)

    def test_illustrative_population_never_invents_data_hashes(self):
        report = prepare(ROOT, PLAN, illustrative_images=40000)
        self.assertEqual(report['schedule']['max_updates'], 250000)
        self.assertEqual(report['data_identity']['basis'], 'illustrative_arithmetic_only')
        self.assertIsNone(report['train_config_preparation']['image_manifest_sha256'])
        self.assertFalse(report['ready_for_training'])

    def test_invalid_schedule_inputs_and_missing_eval_checkpoint(self):
        for value in (True, 0, -1, 2.5, '40000'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                epoch_schedule(value, RECIPE)
        bad = copy.deepcopy(RECIPE)
        bad['checkpoint_every_epochs'] = 3
        with self.assertRaisesRegex(ValueError, 'saved checkpoint'):
            epoch_schedule(40000, bad)

    def write_fixture(self, directory, change=None):
        manifest = {'schema_version': 3, 'illustrative_only': False,
                    'images': [{'image_id': i, 'source_dataset': 'fashionpedia', 'split': 'train'}
                               for i in range(1, 257)]}
        coco = {'images': [{'id': i} for i in range(1, 257)],
                'categories': copy.deepcopy(CATEGORIES),
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1}]}
        if change:
            change(manifest, coco)
        paths = Path(directory) / 'image_manifest.json', Path(directory) / 'annotations_coco.json'
        for path, data in zip(paths, (manifest, coco)):
            path.write_text(json.dumps(data))
        return paths

    def test_metadata_binding_is_not_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, annotations = self.write_fixture(tmp)
            report = prepare(ROOT, PLAN, manifest=manifest, annotations=annotations)
            self.assertEqual(report['data_identity']['images'], 256)
            self.assertEqual(report['train_config_preparation']['annotation_sha256'], file_hash(annotations))
            self.assertEqual(report['data_identity']['basis'], 'metadata_only_not_acceptance_or_freeze')
            self.assertFalse(report['ready_for_training'])
            self.assertEqual(report['operations']['image_reads'], 0)

    def test_invalid_population_and_category_identity_rejected(self):
        mutations = [
            lambda m, c: m['images'][0].update(split='dev'),
            lambda m, c: m['images'][0].update(source_dataset='deepfashion2'),
            lambda m, c: m['images'][0].update(image_id=2),
            lambda m, c: c['images'].pop(),
            lambda m, c: c['categories'][0].update(name='dress'),
            lambda m, c: c['annotations'][0].update(image_id=999),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                paths = self.write_fixture(tmp, mutate)
                with self.assertRaises(ValueError):
                    metadata_identity(*paths, CATEGORIES)

    def test_pinned_running_configuration_drift_is_rejected(self):
        plan = read_json(PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ref in list(plan['references'].values()) + plan['protected_existing_files']:
                target = root / ref['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / ref['path'], target)
            modified = root / plan['protected_existing_files'][0]['path']
            modified.write_text(modified.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Changed pinned file'):
                prepare(root, PLAN)

    def test_ambiguous_binding_and_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Supply both'):
            prepare(ROOT, PLAN, manifest='unused')
        with self.assertRaisesRegex(ValueError, 'Cannot mix'):
            prepare(ROOT, PLAN, manifest='unused', annotations='unused', illustrative_images=40000)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'prepared'
            report = prepare(ROOT, PLAN)
            write_preparation(output, report)
            before = (output / 'preparation.json').read_bytes()
            with self.assertRaises(FileExistsError):
                write_preparation(output, report)
            self.assertEqual((output / 'preparation.json').read_bytes(), before)

    def test_cli_has_no_execution_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'must_not_exist'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_full_fashionpedia.py'),
                                     '--execute', '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('unrecognized arguments: --execute', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
