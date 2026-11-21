"""Only new reuse, resolution and paired boundary diagnostics; no model smoke."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_resolution_trial import inner_boundary, boundary_iou, paired
from run_resolution_trial import validate_configs
from fashion_grounding.segmentation.evaluation import reuse_eval_package, PROTOCOL
from fashion_grounding.data.schema import sha256


class ResolutionTests(unittest.TestCase):
    def test_boundary_at_image_edge_hole_and_thin_object(self):
        full=np.ones((5,5),bool)
        self.assertEqual(int(inner_boundary(full).sum()),16)
        self.assertEqual(boundary_iou(full,full),1)
        self.assertEqual(boundary_iou(full,np.zeros_like(full)),0)
        hole=full.copy();hole[2,2]=False
        self.assertEqual(int(inner_boundary(hole).sum()),24)
        self.assertAlmostEqual(boundary_iou(full,hole),16/24)
        thin=np.zeros_like(full);thin[2,:]=True
        np.testing.assert_array_equal(inner_boundary(thin),thin)

    def test_paired_losses_are_kept_and_changed_population_rejected(self):
        def row(i,m50,m75):
            return dict(annotation_id=i,image_id=i,category_id=6,area=10,
                        matched50=m50,matched75=m75,inner_boundary_iou_1px=.2,
                        extent=dict(precision=.7,recall=.8,false_positive_pixels=2,false_negative_pixels=1))
        a=[row(1,True,False),row(2,False,False),row(3,True,True)]
        b=[row(1,False,False),row(2,True,True),row(3,True,True)]
        r=paired(a,b)
        self.assertEqual(r['matched50']['net'],0)
        self.assertEqual(r['matched50']['gained'],[2]);self.assertEqual(r['matched50']['lost'],[1])
        self.assertEqual(r['matched75']['net'],1)
        self.assertEqual(r['common_matched50_ids'],[3])
        self.assertEqual(r['boundary_all_small']['n'],3)
        with self.assertRaises(ValueError):paired(a,b[:-1])
        with self.assertRaises(ValueError):paired(a,b+[b[0]])

    def test_resize_only_and_protocol_change_rejection(self):
        old=dict(short_edge=800,max_size=1333,max_detections=100,amp=False,budget_seconds=1800)
        new=dict(old,short_edge=1024,max_size=1706,budget_seconds=700)
        validate_configs(old,new)
        for bad in (dict(new,max_detections=200),dict(new,amp=True),dict(new,max_size=1333)):
            with self.assertRaises(ValueError):validate_configs(old,bad)

    def test_reuse_receipt_has_no_image_or_mask_dependency_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dev=root/'dev';train=root/'train';dev.mkdir();train.mkdir()
            for d,split in ((dev,'dev'),(train,'train')):
                (d/'image_manifest.json').write_text(json.dumps(dict(images=[dict(split=split)])))
                (d/'annotations_coco.json').write_text(json.dumps(dict(images=[dict(id=1)],annotations=[])))
            ident=lambda d:dict(image_manifest_sha256=sha256(d/'image_manifest.json'),annotation_sha256=sha256(d/'annotations_coco.json'))
            r=dict(purpose='dev_checkpoint_selection',protocol=PROTOCOL,complete=True,selection_eligible=True,
                   inference_errors=0,evaluated_images=1,evaluation_data=ident(dev),training_data=ident(train),overlap=dict(image_id=0))
            path=root/'report.json';path.write_text(json.dumps(r));cfg=dict(purpose=r['purpose'],**ident(dev))
            _,_,receipt=reuse_eval_package(dev,cfg,train,path)
            self.assertFalse(receipt['image_and_mask_reaudit'])
            r['complete']=False;path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):reuse_eval_package(dev,cfg,train,path)
            r['complete']=True;path.write_text(json.dumps(r));(train/'annotations_coco.json').write_text('{}')
            with self.assertRaises(ValueError):reuse_eval_package(dev,cfg,train,path)

if __name__=='__main__':unittest.main()
