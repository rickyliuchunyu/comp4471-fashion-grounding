"""Regression for binary instance padding, independent of any model/weights."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from fashion_grounding.data.schema import encode_mask


class BinaryInstancePaddingTests(unittest.TestCase):
    def setUp(self):
        try:
            import detectron2  # noqa: F401
        except ImportError:
            self.skipTest('Uses the existing segmentation server environment')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'square.png'
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(self.path)
        self.config = dict(image_size=32, min_scale=.25, max_scale=.25, random_flip='none')

    def mapped(self, masks):
        from fashion_grounding.segmentation.data import map_training_record
        record = dict(file_name=str(self.path), image_id=1, height=32, width=32,
                      annotations=[dict(category_id=i, segmentation=encode_mask(mask))
                                   for i, mask in enumerate(masks)])
        return map_training_record(record, self.config)

    def test_downscaled_binary_mask_has_zero_padding(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        result = self.mapped([mask])
        # 32 -> 8 nearest-neighbour resize: this exact square becomes [2:6,2:6].
        expected = torch.zeros((1, 32, 32), dtype=torch.bool)
        expected[0, 2:6, 2:6] = True
        self.assertTrue(torch.equal(result['instances'].gt_masks, expected))
        self.assertEqual(result['instances'].gt_boxes.tensor.tolist(), [[2.,2.,6.,6.]])
        self.assertTrue((result['image'][:,8:,:] == 128).all())

    def test_object_lost_during_resize_is_not_resurrected_by_padding(self):
        mask = np.zeros((32,32),dtype=bool)
        mask[31,31] = True
        # Nearest-neighbour 4x downsampling samples 2,6,...,30; pixel31 disappears.
        result = self.mapped([mask])
        self.assertEqual(len(result['instances']), 0)
        self.assertEqual(tuple(result['instances'].gt_masks.shape), (0,32,32))


if __name__ == '__main__':
    unittest.main()
