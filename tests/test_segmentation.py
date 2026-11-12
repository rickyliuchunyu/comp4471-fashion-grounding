"""Hand-computable decoder/COCO serialization and strict head-loading checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_api
import torch
from torch import nn

from fashion_grounding.segmentation.predict import decode_candidates, export_package, sha256, write_json
from fashion_grounding.segmentation.model import adapt_coco_state, CLASSIFIER_KEYS


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.classes = torch.full((4, 9), -10.)
        self.classes[0, 6] = 10  # project bag=7
        self.classes[1, 6] = 10
        self.classes[2, 0] = 10
        self.classes[3, 8] = 20  # no-object suppresses foreground probability
        self.masks = torch.full((4, 3, 4), -10.)
        self.masks[0, 1:, 1:3] = 10
        self.masks[1, 1:, 1:3] = 10  # same mask remains: no NMS
        self.masks[3, 0, 0] = 10

    def test_foreground_score_empty_background_and_ties(self):
        result = decode_candidates(self.classes, self.masks, 21)
        self.assertEqual([c.decoder_query_id for c in result], [0, 1])
        self.assertEqual([c.category_id for c in result], [7, 7])
        self.assertEqual(result[0].bbox_xyxy, (1, 1, 3, 3))
        expected = float(self.classes.softmax(-1)[0, 6] * torch.sigmoid(torch.tensor(10.)) * 4 / (4+1e-6))
        self.assertAlmostEqual(result[0].score, expected, places=6)
        self.assertEqual(len(decode_candidates(self.classes, self.masks, 21, max_candidates=1)), 1)

    def test_no_candidates_and_zero_logit(self):
        self.assertEqual(decode_candidates(self.classes, torch.zeros_like(self.masks), 1), [])
        self.assertEqual(decode_candidates(self.classes[:0], self.masks[:0], 1), [])

    def test_invalid_outputs_fail_not_empty(self):
        for classes, masks in [(self.classes[:, :8], self.masks),
                               (self.classes * float('nan'), self.masks),
                               (self.classes, self.masks * float('inf')),
                               (self.classes, self.masks[:1])]:
            with self.assertRaises(ValueError):
                decode_candidates(classes, masks, 1)

    def test_one_foreground_class_even_when_no_object_wins(self):
        classes = torch.zeros((1, 9))
        classes[0, 8] = 1
        result = decode_candidates(classes, torch.ones((1, 1, 1)), 1, score_threshold=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category_id, 1)
        self.assertLess(result[0].score, 0.1)

    def test_package_roundtrip_and_error_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pixel.ppm').write_bytes(b'P6\n4 3\n255\n'+bytes(36))
            images = [dict(image_id=i, file_name='pixel.ppm', width=4, height=3,
                           sha256=sha256(root/'pixel.ppm'), split='fixture', source_dataset='synthetic')
                      for i in (21, 22, 23)]
            manifest = root / 'image_manifest.json'
            write_json(manifest, dict(schema_version=3, illustrative_only=True,
                                     manifest_id='synthetic', mapping_version='whole8-v1-draft', images=images))
            predictions = {21: decode_candidates(self.classes, self.masks, 21), 22: [],
                           23: dict(code='MODEL_ERROR', message='Synthetic failure')}
            kwargs = dict(candidate_set_id='synthetic-test', producer_run_id='unit-test',
                          checkpoint_sha256='a'*64, decoder_config_sha256='b'*64,
                          illustrative_only=True)
            output = root / 'out'
            package = export_package(output, manifest, predictions, **kwargs)
            self.assertEqual([r['status'] for r in package['images']], ['ok', 'no_candidates', 'inference_error'])
            for c, expected in zip(package['images'][0]['candidates'], predictions[21]):
                rle = json.loads((output/c['mask_reference']).read_text())
                np.testing.assert_array_equal(mask_api.decode(rle), expected.mask)
                self.assertEqual(c['mask_sha256'], sha256(output/c['mask_reference']))
                self.assertIsNone(c['source_annotation_id'])
            with self.assertRaises(FileExistsError):
                export_package(output, manifest, predictions, **kwargs)
            with self.assertRaises(ValueError):
                export_package(root/'missing', manifest, {21: []}, **kwargs)
            with self.assertRaises(ValueError):
                export_package(root/'relabel', manifest, predictions, **{**kwargs, 'illustrative_only': False})

    def test_rle_column_major_and_edge_bbox(self):
        logits = torch.full((1, 3, 4), -1.)
        logits[0, 0, 3] = 10
        logits[0, 2, 0] = 10
        result = decode_candidates(self.classes[:1], logits, 1)
        self.assertEqual(result[0].bbox_xyxy, (0, 0, 4, 3))
        encoded = mask_api.encode(np.asfortranarray(result[0].mask, dtype=np.uint8))
        np.testing.assert_array_equal(mask_api.decode(encoded), result[0].mask)

    def test_real_images_diagnostic_manifest_matches_consumer_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'image.ppm').write_bytes(b'P6\n4 3\n255\n'+bytes(36))
            source = root/'image_manifest.json'
            write_json(source, dict(schema_version=3, illustrative_only=False,
                images=[dict(image_id=21, file_name='image.ppm', width=4, height=3,
                             sha256=sha256(root/'image.ppm'))]))
            original_hash = sha256(source)
            pack = export_package(root/'output', source, {21: []}, candidate_set_id='diagnostic',
                producer_run_id='unit-test', checkpoint_sha256='a'*64, decoder_config_sha256='b'*64,
                illustrative_only=True)
            exported = root/'output/image_manifest.json'
            self.assertTrue(json.loads(exported.read_text())['illustrative_only'])
            self.assertEqual(pack['image_manifest']['sha256'], sha256(exported))
            self.assertEqual(pack['provenance']['source_image_manifest_sha256'], original_hash)
            self.assertEqual(sha256(source), original_hash)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.sem_seg_head = nn.Module()
        self.model.sem_seg_head.predictor = nn.Module()
        self.model.sem_seg_head.predictor.class_embed = nn.Linear(2, 9)
        self.model.backbone = nn.Linear(2, 2)
        self.state = copy.deepcopy(self.model.state_dict())
        self.state['sem_seg_head.predictor.class_embed.weight'] = torch.ones(81, 2)
        self.state['sem_seg_head.predictor.class_embed.bias'] = torch.ones(81)

    def test_only_classifier_skipped(self):
        head = self.model.sem_seg_head.predictor.class_embed.weight.detach().clone()
        self.state['backbone.weight'].fill_(3)
        report = adapt_coco_state(self.model, self.state)
        self.assertEqual(set(report['skipped_classifier_keys']), CLASSIFIER_KEYS)
        self.assertTrue(torch.equal(head, self.model.sem_seg_head.predictor.class_embed.weight))
        self.assertTrue(torch.all(self.model.backbone.weight == 3))

    def test_missing_extra_wrong_shape_rejected(self):
        for mutation in ('missing', 'extra', 'shape', 'wrong_head'):
            state = copy.deepcopy(self.state)
            if mutation == 'missing': del state['backbone.bias']
            if mutation == 'extra': state['unknown'] = torch.ones(1)
            if mutation == 'shape': state['backbone.weight'] = torch.ones(3, 2)
            if mutation == 'wrong_head': state['sem_seg_head.predictor.class_embed.bias'] = torch.ones(8)
            with self.assertRaises(ValueError): adapt_coco_state(self.model, state)


if __name__ == '__main__':
    unittest.main()
