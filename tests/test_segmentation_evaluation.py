"""Hand-computable AP, split leakage and deterministic selection boundaries."""
import copy
from pathlib import Path
import unittest

import numpy as np

from fashion_grounding.data.schema import encode_mask
from fashion_grounding.segmentation.evaluation import (
    PROTOCOL, evaluate_masks, choose_checkpoint, check_partition, read_eval_config)


class APTests(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        self.rle = encode_mask(mask)
        self.coco = dict(images=[dict(id=1, width=4, height=4)],
            categories=[dict(id=i, name=str(i)) for i in range(1, 9)],
            annotations=[dict(id=1, image_id=1, category_id=1, iscrowd=0,
                              segmentation=self.rle, bbox=[0, 0, 2, 2], area=4)])
        self.prediction = dict(image_id=1, category_id=1, score=0.9, segmentation=self.rle)

    def score(self, detections):
        return evaluate_masks(self.coco, [dict(image_id=1, status='ok' if detections else 'no_detections', detections=detections)])

    def test_perfect_and_absent_categories_are_na(self):
        report = self.score([self.prediction])
        for key in ('AP', 'AP50', 'AP75', 'AP_small'):
            self.assertAlmostEqual(report['metrics'][key], 100)
        self.assertIsNone(report['metrics']['per_category_AP']['8'])
        self.assertIsNone(report['metrics']['AP_large'])

    def test_empty_wrong_class_and_empty_upstream_mask(self):
        self.assertEqual(self.score([])['metrics']['AP'], 0)
        self.assertEqual(self.score([{**self.prediction, 'category_id': 2}])['metrics']['AP'], 0)
        empty = encode_mask(np.zeros((4, 4), dtype=bool))
        self.assertEqual(self.score([{**self.prediction, 'segmentation': empty, 'score': 0.0}])['metrics']['AP'], 0)

    def test_two_gt_and_duplicate_before_second_match(self):
        self.coco['images'].append(dict(id=2, width=4, height=4))
        self.coco['annotations'].append({**self.coco['annotations'][0], 'id': 2, 'image_id': 2})
        rows = [dict(image_id=1, status='ok', detections=[self.prediction, {**self.prediction, 'score': 0.8}]),
                dict(image_id=2, status='ok', detections=[{**self.prediction, 'image_id': 2, 'score': 0.7}])]
        report = evaluate_masks(self.coco, rows)
        self.assertAlmostEqual(report['metrics']['AP'], (51+50*2/3)/101*100)
        with self.assertRaisesRegex(ValueError, 'Every GT image'):
            evaluate_masks(self.coco, rows[:1])

    def test_failure_is_in_denominator_and_not_complete(self):
        report = evaluate_masks(self.coco, [dict(image_id=1, status='inference_error', detections=[], error='synthetic')])
        self.assertEqual(report['metrics']['AP'], 0)
        self.assertEqual(report['evaluated_images'], 1)
        self.assertEqual(report['inference_errors'], 1)
        self.assertFalse(report['complete'])

    def test_crowd_only_no_positive_is_na(self):
        self.coco['annotations'][0]['iscrowd'] = 1
        self.assertIsNone(self.score([self.prediction])['metrics']['AP'])

    def test_prediction_box_does_not_set_mask_area(self):
        self.assertAlmostEqual(self.score([{**self.prediction, 'bbox': [0, 0, 100, 100]}])['metrics']['AP_small'], 100)

    def test_unknown_duplicate_nonfinite_rows_rejected(self):
        good = dict(image_id=1, status='ok', detections=[self.prediction])
        for rows in ([good, good], [{**good, 'image_id': 2}],
                     [{**good, 'detections': [{**self.prediction, 'score': float('nan')}]}]):
            with self.assertRaises(ValueError):
                evaluate_masks(self.coco, rows)


class SelectionTests(unittest.TestCase):
    def report(self, update, ap):
        return dict(purpose='dev_checkpoint_selection', selection_eligible=True, complete=True,
            inference_errors=0, protocol=PROTOCOL, metric_units='percent', evaluation_config_sha256='config',
            evaluation_data={'image_manifest_sha256':'dev'}, training_data={'image_manifest_sha256':'train'},
            training_config_sha256='training', training_git_commit='source', evaluation_git_commit='eval-source',
            metrics={'AP':ap}, completed_updates=update, checkpoint_sha256=str(update), checkpoint_path=f'/checkpoint-{update}')

    def test_highest_ap_then_earliest_not_argument_order(self):
        reports = [self.report(300, 25), self.report(100, 25), self.report(200, 20)]
        self.assertEqual(choose_checkpoint(reports)['completed_updates'], 100)
        reports[-1]['metrics']['AP'] = 26
        self.assertEqual(choose_checkpoint(reports)['completed_updates'], 200)

    def test_diagnostics_failures_incomparable_and_duplicates_rejected(self):
        for changes in (dict(purpose='pilot_ap_integration'), dict(selection_eligible=False),
                        dict(complete=False), dict(inference_errors=1), dict(evaluation_data={'dev':'other'}),
                        dict(evaluation_config_sha256='other'), dict(training_config_sha256='other'),
                        dict(evaluation_git_commit='other'), dict(completed_updates=100), dict(metrics={'AP':None}),
                        dict(metrics={'AP':float('nan')})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                choose_checkpoint([self.report(100, 10), {**self.report(200, 20), **changes}])


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.train = dict(schema_version=3, illustrative_only=False, images=[dict(image_id=1,
            sha256='a'*64, group_id='train-group', source_image_id='train-1', source_dataset='fashionpedia', split='train')])
        self.dev = copy.deepcopy(self.train)
        self.dev['images'][0].update(image_id=2, sha256='b'*64, group_id='dev-group', source_image_id='dev-2', split='dev')

    def test_disjoint_dev_and_diagnostic_train(self):
        self.assertFalse(any(check_partition(self.dev, self.train, False).values()))
        self.assertEqual(check_partition(self.train, self.train, True)['image_id'], 1)

    def test_each_overlap_and_test_and_wrong_split_fail(self):
        for key in ('image_id', 'sha256', 'group_id', 'source_image_id'):
            dev = copy.deepcopy(self.dev)
            dev['images'][0][key] = self.train['images'][0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overlap'):
                check_partition(dev, self.train, False)
        for split in ('train', 'test', 'unused'):
            self.dev['images'][0]['split'] = split
            with self.assertRaises(ValueError):
                check_partition(self.dev, self.train, False)
        with self.assertRaises(ValueError):
            check_partition(self.dev, self.train, True)

    def test_formal_template_requires_pinned_data(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, 'Pin'):
            read_eval_config(root/'configs/runs/segmentation/dev_ap.json')
        self.assertEqual(read_eval_config(root/'configs/runs/segmentation/pilot_ap_integration.json')['purpose'], 'pilot_ap_integration')


if __name__ == '__main__':
    unittest.main()
