import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from fashion_grounding.segmentation.training_state import (read_config, build_optimizer,
    UpdateSchedule, StatefulSampler, capture_rng, restore_rng, state_digest, save_checkpoint, load_checkpoint)


CONFIG = Path(__file__).resolve().parents[1] / 'configs/runs/segmentation/pilot_stability_200.json'


class TrainingStateTests(unittest.TestCase):
    def test_profile_bounds_and_unpinned_formal_data(self):
        c = read_config(CONFIG)
        self.assertEqual((c['max_updates'], c['image_size'], c['effective_batch_images']), (200, 768, 2))
        with self.assertRaisesRegex(ValueError, 'Pin the real training data'):
            read_config(CONFIG.with_name('s0_train.json'))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'invalid.json'; c['accumulation_steps'] = 2; p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'Effective batch'): read_config(p)

    def test_warmup_and_milestones_at_update_boundaries(self):
        config = {**read_config(CONFIG), 'warmup_updates': 2, 'warmup_start_factor': 0.1,
                  'lr_milestones': [3, 5], 'lr_decay': 0.1}
        optimizer = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=1.)
        scheduler = UpdateSchedule(optimizer, config)
        observed = [optimizer.param_groups[0]['lr']]
        for _ in range(5): scheduler.step(); observed.append(optimizer.param_groups[0]['lr'])
        np.testing.assert_allclose(observed, [0.1, 0.55, 1., 0.1, 0.1, 0.01])

    def test_optimizer_groups_follow_upstream_policy(self):
        model = nn.Module(); model.backbone = nn.Linear(2, 2)
        model.norm = nn.LayerNorm(2); model.embedding = nn.Embedding(2, 2)
        config = read_config(CONFIG); optimizer = build_optimizer(model, config)
        groups = {g['parameter_name']:g for g in optimizer.param_groups}
        self.assertAlmostEqual(groups['backbone.weight']['lr'], 1e-5)
        self.assertEqual(groups['backbone.weight']['weight_decay'], 0.05)
        self.assertEqual(groups['norm.weight']['weight_decay'], 0.)
        self.assertEqual(groups['embedding.weight']['weight_decay'], 0.)

    def test_resume_matches_uninterrupted_updates_and_rng_across_epoch(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        config = read_config(CONFIG)
        model = nn.Linear(2, 1); opt = build_optimizer(model, config)
        schedule = UpdateSchedule(opt, config); sampler = StatefulSampler(3, 42)
        def step(model, opt, schedule, sampler):
            index = sampler.next()
            # Exercise Python/numpy/torch RNG as well as shuffled epoch transitions.
            x = torch.randn(1, 2) + random.random() + np.random.rand()
            opt.zero_grad(); loss = (model(x)-index).square().mean(); loss.backward()
            opt.step(); schedule.step(); return (index, float(loss.detach()))
        for _ in range(2): step(model, opt, schedule, sampler)
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=schedule.state_dict(), sampler=sampler.state_dict(), rng=capture_rng(),
            completed_updates=2, config_sha256='a'*64, data={'hash':'b'*64}))
        expected = [step(model, opt, schedule, sampler) for _ in range(5)]
        final_digest = state_digest(model.state_dict())
        fresh = nn.Linear(2, 1); fresh_opt = build_optimizer(fresh, config)
        fresh_schedule = UpdateSchedule(fresh_opt, config); fresh_sampler = StatefulSampler(3, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth'; save_checkpoint(path, state)
            saved=load_checkpoint(path, 'a'*64, {'hash':'b'*64})
            with self.assertRaises(FileExistsError): save_checkpoint(path, state)
            with self.assertRaisesRegex(ValueError, 'configuration or dataset'): load_checkpoint(path, 'c'*64, {'hash':'b'*64})
            fresh.load_state_dict(saved['model']); fresh_opt.load_state_dict(saved['optimizer'])
            fresh_schedule.load_state_dict(saved['scheduler']); fresh_sampler.load_state_dict(saved['sampler'])
            restore_rng(saved['rng'])
            actual = [step(fresh, fresh_opt, fresh_schedule, fresh_sampler) for _ in range(5)]
            self.assertEqual(expected, actual)
            self.assertEqual(final_digest, state_digest(fresh.state_dict()))
            path.write_bytes(path.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_checkpoint(path, 'a'*64, {'hash':'b'*64})


class LSJTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import detectron2
        except ImportError: raise unittest.SkipTest('Detectron2 LSJ tests run in the segmentation server environment')

    def test_identity_geometry_empty_image_and_rng_replay(self):
        from PIL import Image
        from pycocotools import mask as mask_api
        from fashion_grounding.segmentation.data import map_training_record
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.png'
            pixels = np.zeros((32, 32, 3), dtype=np.uint8); pixels[5:12, 8:19, 0] = 255
            Image.fromarray(pixels).save(path)
            mask = pixels[:,:,0] > 0
            record = dict(file_name=str(path), width=32, height=32, image_id=1,
                          annotations=[dict(segmentation=mask_api.encode(np.asfortranarray(mask)), category_id=3)])
            config = dict(image_size=32, min_scale=1., max_scale=1., random_flip='none')
            result = map_training_record(record, config)
            np.testing.assert_array_equal(result['instances'].gt_masks[0].numpy(), mask)
            self.assertEqual(result['instances'].gt_boxes.tensor.tolist(), [[8.,5.,19.,12.]])
            self.assertEqual(result['instances'].gt_classes.tolist(), [3])
            empty = map_training_record({**record, 'annotations':[]}, config)
            self.assertEqual(tuple(empty['instances'].gt_masks.shape), (0,32,32))
            config.update(min_scale=0.1, max_scale=2., random_flip='horizontal')
            state=capture_rng(); a=map_training_record(record,config);restore_rng(state);b=map_training_record(record,config)
            self.assertTrue(torch.equal(a['image'],b['image']))
            self.assertTrue(torch.equal(a['instances'].gt_masks,b['instances'].gt_masks))


if __name__ == '__main__': unittest.main()
