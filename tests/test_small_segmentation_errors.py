"""Only new diagnostic attribution boundaries; no model or data smoke."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_small_segmentation_errors import extent, geometry_bucket

class SmallDiagnosticsTests(unittest.TestCase):
    def test_exclusive_partition_and_threshold_edges(self):
        self.assertEqual(geometry_bucket(.75,.9),'same_iou_ge_075')
        self.assertEqual(geometry_bucket(.5,.9),'same_iou_050_075')
        self.assertEqual(geometry_bucket(.49,.5),'wrong_class_only_iou_ge_050')
        self.assertEqual(geometry_bucket(.1,.09),'partial_or_mixed_iou_010_050')
        self.assertEqual(geometry_bucket(.09,.09),'no_saved_mask_iou_ge_010')
    def test_extent_distinguishes_missing_pixels_and_spill(self):
        gt=np.zeros((4,4),bool);gt[1:3,1:3]=True
        under=np.zeros_like(gt);under[1,1:3]=True
        a=extent(gt,under);self.assertEqual((a['iou'],a['precision'],a['recall']),(.5,1,.5))
        over=np.ones_like(gt);b=extent(gt,over)
        self.assertEqual((b['iou'],b['precision'],b['recall']),(.25,.25,1))
        empty=extent(gt,np.zeros_like(gt));self.assertEqual(empty['iou'],0);self.assertIsNone(empty['precision'])
        self.assertEqual(extent(gt,gt)['iou'],1)
if __name__=='__main__':unittest.main()
