"""Only new combined-recipe paths. All examples run on CPU without a segmenter."""
from pathlib import Path
import json
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from fashion_grounding.data.schema import encode_mask, sha256, write_json
from fashion_grounding.segmentation.candidate_cache import CacheWriter, export_cached, validate_row
from fashion_grounding.segmentation.data import map_training_record
from fashion_grounding.segmentation.export_candidates import bind_selected_config, validate_config
from fashion_grounding.segmentation.point_observations import observe_final_points, point_hits
from fashion_grounding.segmentation.predict import decode_candidates
from fashion_grounding.segmentation.training_state import read_config, update_image_size

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/runs/segmentation'
VENDOR = ROOT.parents[1]/'vendor/segmentation/Mask2Former'
sys.path.insert(0, str(ROOT/'scripts'))
from full_single_plan import select_full
from run_tenk_adaptation import validate


class ComboTests(unittest.TestCase):
    def test_plan_and_equal_update_number_selection(self):
        plan = json.loads((CONFIG/'full_combo_plan.json').read_text())
        prepared, incumbent, context = validate(plan, ROOT)
        self.assertFalse(prepared['execution_enabled'])
        self.assertEqual(prepared['total_hard_seconds'], 8*3600)
        first, last = prepared['stages']
        self.assertIn('--model-only', first['train'])
        self.assertIn('--resume', last['train'])
        self.assertEqual([s['new_run_update'] for s in (first,last)], [14159,28318])
        self.assertEqual(context['model_only_source']['sha256'], incumbent['checkpoint_sha256'])
        self.assertEqual(first['train'][-1], str(Path(plan['execution_checkout'])/plan['model_only_source']))
        half = dict(incumbent, completed_updates=14159, checkpoint_sha256='a'*64)
        end = dict(incumbent, checkpoint_sha256='b'*64)
        tie = select_full(incumbent, [half,end], [14159,28318], 'full_combo')
        self.assertEqual(tie['selected_origin'], 'cached_28318')
        end['metrics'] = dict(incumbent['metrics'], AP=incumbent['metrics']['AP']+.1)
        selected = select_full(incumbent, [half,end], [14159,28318], 'full_combo')
        template = validate_config(json.loads((CONFIG/'candidate_u28318_components.json').read_text()))
        self.assertEqual(template['source_decoder_sha256'], sha256(ROOT/'src/fashion_grounding/segmentation/predict.py'))
        self.assertEqual(bind_selected_config(template, selected, end)['checkpoint_sha256'], 'b'*64)
        with self.assertRaises(ValueError):
            bind_selected_config(template, selected, incumbent)

    def test_scale_pairs_and_sampled_updates_survive_split_without_rng(self):
        cfg = read_config(CONFIG/'full_combo_train.json')
        rng = random.getstate()
        sizes = [update_image_size(cfg, u) for u in range(1, 28319)]
        self.assertEqual(sizes.count(896), 14159)
        for left,right in zip(sizes[::2],sizes[1::2]):
            self.assertEqual({left,right}, {768,896})
        observed = [s for i,s in enumerate(sizes) if i % cfg['point_stats_interval'] < 2]
        self.assertEqual(observed.count(768), observed.count(896))
        self.assertEqual(update_image_size(cfg, 14160), sizes[14159])
        self.assertEqual(random.getstate(), rng)

    def test_point_counts_have_known_geometry(self):
        mask = np.zeros((5,5), bool)
        mask[1:4,1:4] = True
        xy = np.array([[.5,.5],[.3,.5],[.1,.5],[.9,.9]])
        actual = point_hits(mask, xy, np.array([1.,1.,0.,0.]))
        self.assertEqual(actual, dict(points=4, foreground_hits=2, boundary_hits=3, augmented_area=9))

    def test_score_components_survive_cache_and_threshold_projection(self):
        logits = torch.zeros(100,9)
        logits[17,4] = 3
        masks = torch.full((100,4,4),-2.)
        masks[17,1:3,:] = 2.
        c, = decode_candidates(logits, masks, 7, 0)
        self.assertEqual(c.decoder_query_id,17)
        self.assertAlmostEqual(c.class_probability, np.exp(3)/(8+np.exp(3)), places=6)
        self.assertAlmostEqual(c.mask_quality, 1/(1+np.exp(-2))*8/(8+1e-6), places=6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'photo.png'
            Image.new('RGB',(4,4)).save(path)
            im = dict(image_id=7, file_name=path.name, sha256=sha256(path), split='dev', width=4, height=4)
            manifest = root/'images.json'
            write_json(manifest, dict(schema_version=3, illustrative_only=True, images=[im]))
            cfg = json.loads((CONFIG/'candidate_u28318_components.json').read_text())
            writer = CacheWriter(root/'cache', manifest, dict(producer_run_id='fixture',
                checkpoint_sha256='1'*64, score_components=cfg['score_components']), cfg['decoder'])
            writer.append(7,[c],dict(class_logits_shape=[100,9], mask_logits_original_shape=[100,4,4],
                all_logits_finite=True, candidate_count=1))
            writer.finish()
            result = export_cached(writer.root,root/'trial',threshold=.3,candidate_set_id='fixture',
                                   expected_checkpoint_sha256='1'*64)
            entry, = result['images'][0]['candidates']
            self.assertEqual((entry['class_probability'],entry['mask_quality'],entry['score'],entry['decoder_query_id']),
                             (c.class_probability,c.mask_quality,c.score,17))
            row = json.loads((writer.root/'records/7.json').read_text())
            del row['candidates'][0]['mask_quality']
            with self.assertRaisesRegex(ValueError, 'score components'):
                validate_row(row, im, writer.root, components_required=True)
            del row['candidates'][0]['class_probability']
            validate_row(row, im, writer.root)


@unittest.skipUnless(VENDOR.is_dir(), 'requires pinned segmentation vendor on project server')
class UpstreamObservationTests(unittest.TestCase):
    def test_mapper_preserves_identity_at_both_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'in.png'
            Image.new('RGB',(32,32)).save(path)
            masks = [np.zeros((32,32),bool), np.eye(32,dtype=bool)]
            record = dict(file_name=str(path), image_id=3, height=32, width=32,
                annotations=[dict(annotation_id=i, category_id=7, source_category_name='sock',
                                  source_area=int(m.sum()),segmentation=encode_mask(m)) for i,m in enumerate(masks)])
            for size in (768,896):
                item = map_training_record(record, dict(image_size=size, min_scale=1.,max_scale=1.,
                          random_flip='none'), observe_points=True)
                self.assertEqual(tuple(item['image'].shape), (3,size,size))
                self.assertEqual(item['point_targets'],[dict(annotation_id=1,source_category='sock',original_area=32)])
                self.assertEqual(len(item['instances']),1)

    def test_real_final_loss_points_preserve_auxiliary_losses_and_rng(self):
        sys.path.insert(0, str(VENDOR))
        from mask2former.modeling.criterion import SetCriterion
        criterion = SetCriterion(8, None, {}, .1, ['masks'], 32, 3., .75)
        pred = dict(pred_masks=torch.zeros(1,2,4,4))
        masks = torch.zeros(2,8,8,dtype=torch.bool)
        masks[0,:4,:4] = True
        masks[1,:,:] = True
        gt = [dict(masks=masks)]
        ids = [(torch.tensor([0,1]),torch.tensor([1,0]))]
        batch = [dict(image_id=7,point_targets=[dict(annotation_id=11),dict(annotation_id=12)])]
        def losses():
            return [criterion.loss_masks(pred,gt,ids,2.) for _ in range(2)]
        torch.manual_seed(9)
        expected = losses()
        rng = torch.get_rng_state()
        torch.manual_seed(9)
        rows = []
        with observe_final_points(criterion,batch,rows):
            actual = losses()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual([r['annotation_id'] for r in rows],[12,11])
        self.assertEqual([r['augmented_area'] for r in rows],[64,16])
        self.assertEqual([r['points'] for r in rows],[32,32])
        for left,right in zip(actual,expected):
            for key in right:
                self.assertTrue(torch.equal(left[key],right[key]))


if __name__ == '__main__':
    unittest.main()
