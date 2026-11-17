"""Only the new warmstart/schedule/visibility/crop contracts; no model forward."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from pycocotools import mask as mask_api

from fashion_grounding.segmentation.training_state import (UpdateSchedule, StatefulSampler,
    build_optimizer, config_sha256, read_config, save_checkpoint, state_digest)
from fashion_grounding.segmentation.model_warmstart import load_model_only

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/runs/segmentation/tenk_adaptation_train.json'


class WarmstartTests(unittest.TestCase):
    def test_legacy_multistep_keeps_minimal_contract_and_state_schema(self):
        opt=torch.optim.AdamW(nn.Linear(1,1).parameters(),lr=1.)
        config=dict(warmup_updates=0,warmup_start_factor=1.,lr_milestones=[2],lr_decay=.1)
        schedule=UpdateSchedule(opt,config)
        self.assertEqual(set(schedule.state_dict()),{'base_lrs','warmup','start','milestones','gamma','completed_updates'})
        schedule.step(); self.assertEqual(opt.param_groups[0]['lr'],1.)
        schedule.step(); self.assertAlmostEqual(opt.param_groups[0]['lr'],.1)

    def fixture(self, path):
        model = nn.Linear(2,2)
        with torch.no_grad(): model.weight.fill_(3.); model.bias.fill_(7.)
        state = dict(model=model.state_dict(), completed_updates=20000, config={'seed':42},
            config_sha256=config_sha256({'seed':42}), data={'train':'old'}, git_commit='source',
            optimizer='must not restore', scheduler='must not restore', rng='must not restore',
            component_digests={'model':state_digest(model.state_dict())})
        digest = save_checkpoint(path,state)
        spec = dict(mode='model_only',path=str(path),sha256=digest,source_completed_updates=20000,
                    source_training_data=state['data'],source_config_sha256=state['config_sha256'],source_git_commit='source')
        return model,spec

    def test_only_weights_loaded_with_fresh_optimizer_schedule_sampler_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected,spec = self.fixture(Path(tmp)/'parent.pth')
            model=nn.Linear(2,2); before=torch.get_rng_state().clone()
            receipt=load_model_only(model,spec)
            self.assertEqual(state_digest(model.state_dict()),state_digest(expected.state_dict()))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            config=read_config(CONFIG); opt=build_optimizer(model,config); schedule=UpdateSchedule(opt,config)
            sampler=StatefulSampler(10000,config['seed'])
            self.assertEqual(len(opt.state),0); self.assertEqual(schedule.completed_updates,0)
            self.assertEqual((sampler.position,sampler.epoch),(0,0))
            self.assertEqual(receipt['loaded_components'],['model'])
            self.assertAlmostEqual(opt.param_groups[0]['lr'],5e-8)

    def test_wrong_source_and_incompatible_head_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,spec=self.fixture(Path(tmp)/'parent.pth')
            for key,value in [('sha256','a'*64),('source_completed_updates',18000),('source_training_data',{'train':'new'})]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    load_model_only(nn.Linear(2,2),{**spec,key:value})
            with self.assertRaisesRegex(ValueError,'identical keys, shapes'):
                load_model_only(nn.Linear(2,3),spec)

    def test_cosine_boundaries_and_resume_do_not_restart_warmup(self):
        config={**read_config(CONFIG),'warmup_updates':2,'max_updates':10,'warmup_start_factor':.1,'lr_min_factor':.02}
        opt=torch.optim.AdamW(nn.Linear(1,1).parameters(),lr=1.)
        schedule=UpdateSchedule(opt,config)
        observed=[opt.param_groups[0]['lr']]
        for _ in range(10):
            schedule.step(); observed.append(opt.param_groups[0]['lr'])
            if schedule.completed_updates==6: saved=copy.deepcopy(schedule.state_dict())
        np.testing.assert_allclose([observed[i] for i in [0,1,2,6,10]],[.1,.55,1.,.51,.02])
        newopt=torch.optim.AdamW(nn.Linear(1,1).parameters(),lr=1.)
        restored=UpdateSchedule(newopt,config); restored.load_state_dict(saved)
        self.assertAlmostEqual(newopt.param_groups[0]['lr'],observed[6])
        restored.step(); self.assertAlmostEqual(newopt.param_groups[0]['lr'],observed[7])
        with self.assertRaises(ValueError):
            UpdateSchedule(newopt,{**config,'max_updates':20}).load_state_dict(saved)

    def test_equivalent_epoch_budget_rejects_wrong_batch(self):
        config=json.loads(CONFIG.read_text()); config['training_images']=9999
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'; p.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'equivalent-epoch'): read_config(p)


class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import detectron2
        except ImportError: raise unittest.SkipTest('Geometry cases run on the existing server CPU runtime')

    def test_retention_uses_resized_area_and_records_disappearance(self):
        from detectron2.data import transforms as T
        from fashion_grounding.segmentation.visibility import mask_visibility
        mask=np.ones((8,8),dtype=np.uint8)
        result=mask_visibility(mask,T.TransformList([T.ResizeTransform(8,8,16,16),T.CropTransform(0,0,8,16)]))
        self.assertEqual(result['augmented_area'],128); self.assertEqual(result['retained_fraction'],.5)
        small=np.zeros((8,8),dtype=np.uint8); small[6:,6:]=1
        result=mask_visibility(small,T.TransformList([T.CropTransform(0,0,4,4)]))
        self.assertTrue(result['disappeared']); self.assertEqual(result['retained_fraction'],0.)

    def test_target_geometry_preserves_edge_target_and_all_other_visible_masks(self):
        from fashion_grounding.segmentation.data import map_training_record
        from PIL import Image
        config=read_config(CONFIG); config.update(image_size=64,random_flip='none')
        config['target_crop']=json.loads((CONFIG.parent/'tenk_visibility_target_context.json').read_text())['target_crop']
        image=np.zeros((100,160,3),dtype=np.uint8)
        a=np.zeros((100,160),dtype=np.uint8); a[95:100,155:160]=1
        b=np.zeros_like(a); b[60:100,100:160]=1
        image[a>0,0]=255
        annotations=[dict(annotation_id=i,source_area=int(m.sum()),category_id=i,
            segmentation=mask_api.encode(np.asfortranarray(m))) for i,m in enumerate([a,b])]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.png'; Image.fromarray(image).save(path)
            record=dict(file_name=str(path),height=100,width=160,image_id=1,annotations=annotations)
            with patch('fashion_grounding.segmentation.target_crop.random.random',return_value=0.):
                result=map_training_record(record,config)
            self.assertEqual(result['augmentation_view'],'target_context')
            self.assertEqual(result['instances'].gt_classes.tolist(),[0,1])
            self.assertTrue(result['instances'].gt_masks[0].any())
            self.assertEqual(tuple(result['image'].shape),(3,64,64))
            self.assertLess(int(result['instances'].gt_masks[0].sum()),64*64)
            box=result['instances'].gt_boxes.tensor[0]
            self.assertTrue(bool((box>=0).all() and (box<=64).all()))

    def test_unchosen_crop_keeps_original_lsj_rng_and_pixels(self):
        from detectron2.data import transforms as T
        from fashion_grounding.segmentation.data import training_transforms
        config=read_config(CONFIG); config['image_size']=64
        config['target_crop']=json.loads((CONFIG.parent/'tenk_visibility_target_context.json').read_text())['target_crop']
        image=np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3)
        np.random.seed(7)
        expected,et=T.apply_transform_gens([T.RandomFlip(horizontal=True,vertical=False),
            T.ResizeScale(.5,2.,64,64),T.FixedSizeCrop((64,64),seg_pad_value=0)],image)
        np.random.seed(7)
        with patch('fashion_grounding.segmentation.target_crop.random.random',return_value=.9):
            actual,at=training_transforms(image,config,[])
        np.testing.assert_array_equal(actual,expected)
        mask=(image[:,:,0]%3==0).astype(np.uint8)
        np.testing.assert_array_equal(at.apply_segmentation(mask),et.apply_segmentation(mask))

    def test_diagnostic_cohorts_disjoint_and_train_selection_deterministic(self):
        from fashion_grounding.segmentation.visibility import select_images
        images=[{'image_id':i} for i in range(20)]
        annotations=[dict(image_id=i,area=10,category_id=1) for i in range(20)]
        a=select_images(images,annotations,set(range(10)),42,uniform=5,targeted=7)
        self.assertEqual(a,select_images(images,annotations,set(range(10)),42,uniform=5,targeted=7))
        self.assertEqual(list(a.values()).count('uniform'),5); self.assertEqual(list(a.values()).count('targeted'),7)


class PlanTests(unittest.TestCase):
    def test_planned_phases_reset_only_once_and_never_exceed_ten_epochs(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from prepare_tenk_adaptation import prepare
        plan=json.loads((CONFIG.parent/'tenk_adaptation_plan.json').read_text())
        result=prepare(plan)
        self.assertEqual([s['new_run_update'] for s in result['stages']],[10000,20000,30000,40000,50000])
        self.assertEqual(result['max_image_presentations'],100000)
        self.assertIn('--model-only',result['stages'][0]['train'])
        for stage in result['stages'][1:]:
            self.assertIn('--resume',stage['train']); self.assertNotIn('--model-only',stage['train'])
        self.assertFalse(read_config(CONFIG).get('target_crop'))

    def test_unapproved_execution_and_changed_evaluation_spacing_rejected(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from prepare_tenk_adaptation import prepare
        plan=json.loads((CONFIG.parent/'tenk_adaptation_plan.json').read_text())
        for change in [{'training_authorized':True},{'evaluation_epochs':[2,4,6,8,12]}]:
            with self.subTest(change=change),self.assertRaises(ValueError): prepare({**plan,**change})

    def test_disabled_trainer_stops_before_output_or_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'must-not-exist'
            result=subprocess.run([sys.executable,'-m','fashion_grounding.segmentation.train',
                '--config',str(CONFIG),'--package','nonexistent','--source','nonexistent',
                '--model-only','nonexistent','--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('not authorized for training',result.stderr)
            self.assertFalse(output.exists())


if __name__=='__main__': unittest.main()
