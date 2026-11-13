from collections import OrderedDict
from datetime import date
import hashlib
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fashion_grounding.segmentation.checkpoint import read_coco_checkpoint
from fashion_grounding.segmentation.model import adapt_coco_state, LOSS_CLASS_WEIGHT_KEY


class CheckpointReaderTests(unittest.TestCase):
    def read(self, value, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.pkl'
            payload = pickle.dumps(value, protocol=4)
            path.write_bytes(payload)
            config = dict(expected_sha256=hashlib.sha256(payload).hexdigest(),
                          expected_size=len(payload), md5_prefix=hashlib.md5(payload).hexdigest()[:6])
            return read_coco_checkpoint(path, **{**config, **overrides})

    def test_numeric_model_and_checksum_before_deserialization(self):
        value = dict(model=OrderedDict(weight=np.ones((2, 3), dtype=np.float32)),
                     __author__='MaskFormer Model Zoo')
        state, report = self.read(value)
        np.testing.assert_array_equal(state['weight'], value['model']['weight'])
        self.assertTrue(report['verified_before_deserialization'])
        with patch('fashion_grounding.segmentation.checkpoint._NumpyCheckpointReader') as reader:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.read(value, expected_sha256='0'*64)
            reader.assert_not_called()

    def test_forbidden_global_and_nonfinite_arrays(self):
        with self.assertRaisesRegex(pickle.UnpicklingError, 'Unsupported checkpoint global'):
            self.read(date(2026, 11, 12))
        for bad in (np.array([np.nan], dtype=np.float32), np.array([1], dtype=np.int64)):
            with self.assertRaises(ValueError):
                self.read(dict(model={'weight': bad}, __author__='MaskFormer Model Zoo'))


class ActualFormatAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.sem_seg_head = nn.Module()
        self.model.sem_seg_head.predictor = nn.Module()
        predictor = self.model.sem_seg_head.predictor
        predictor.class_embed = nn.Linear(2, 9)
        predictor.query_feat = nn.Embedding(3, 2)
        self.model.criterion = nn.Module()
        self.model.criterion.register_buffer('empty_weight', torch.tensor([1.]*8+[0.1]))
        self.state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.state['sem_seg_head.predictor.class_embed.weight'] = torch.ones(81, 2)
        self.state['sem_seg_head.predictor.class_embed.bias'] = torch.ones(81)
        self.state[LOSS_CLASS_WEIGHT_KEY] = torch.tensor([1.]*80+[0.1])
        self.state['sem_seg_head.predictor.static_query.weight'] = self.state.pop('sem_seg_head.predictor.query_feat.weight') + 5

    def test_legacy_query_loaded_and_loss_weight_preserved(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        report = adapt_coco_state(self.model, self.state)
        self.assertEqual(report['loaded_tensor_count'], 1)
        self.assertTrue(report['exact_tensor_equality_verified'])
        self.assertEqual(report['retained_loss_buffers'], [LOSS_CLASS_WEIGHT_KEY])
        self.assertTrue(torch.equal(self.model.state_dict()[LOSS_CLASS_WEIGHT_KEY], original[LOSS_CLASS_WEIGHT_KEY]))
        self.assertTrue(torch.equal(self.model.sem_seg_head.predictor.query_feat.weight,
                                    self.state['sem_seg_head.predictor.static_query.weight']))
        self.assertTrue(torch.equal(self.model.sem_seg_head.predictor.class_embed.bias,
                                    original['sem_seg_head.predictor.class_embed.bias']))

    def test_ambiguous_query_wrong_loss_weight_and_dtype_rejected(self):
        for mutation in ('duplicate', 'loss_weight', 'dtype'):
            state = {k: v.clone() for k, v in self.state.items()}
            if mutation == 'duplicate':
                state['sem_seg_head.predictor.query_feat.weight'] = state['sem_seg_head.predictor.static_query.weight']
            if mutation == 'loss_weight': state[LOSS_CLASS_WEIGHT_KEY][-1] = 0.2
            if mutation == 'dtype': state['sem_seg_head.predictor.static_query.weight'] = state['sem_seg_head.predictor.static_query.weight'].double()
            with self.assertRaises(ValueError): adapt_coco_state(self.model, state)


if __name__ == '__main__':
    unittest.main()
