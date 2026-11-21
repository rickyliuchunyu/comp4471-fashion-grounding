"""New cache/export behavior only: synthetic pixels/predictions, no model forward."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from fashion_grounding.data.schema import write_json, read_json, sha256
from fashion_grounding.segmentation.candidate_cache import CacheWriter, export_cached, fingerprint
from fashion_grounding.segmentation.export_candidates import (
    cache_images, input_manifest, validate_checkpoint_metadata, validate_config)


class CandidateExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        images = []
        for iid in (17, 23):
            path = self.source/f'{iid}.png'
            Image.fromarray(np.full((4, 6, 3), iid, dtype=np.uint8)).save(path)
            images.append(dict(image_id=iid, file_name=path.name, width=6, height=4,
                               sha256=sha256(path), split='dev', group_id=str(iid)))
        self.manifest = self.source/'image_manifest.json'
        write_json(self.manifest, dict(schema_version=3, illustrative_only=True, images=images))
        self.config = read_json(Path(__file__).resolve().parents[1]/
                               'configs/runs/segmentation/candidate_u3000_export.json')
        self.checkpoint_hash = self.config['checkpoint_sha256']

    def candidate(self, qid, score):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:5] = True
        return SimpleNamespace(candidate_id=f'image-17-query-{qid:03d}', decoder_query_id=qid,
            score=score, category_id=2, bbox_xyxy=(2, 1, 5, 3), mask=mask)

    def writer(self, name='cache'):
        return CacheWriter(self.root/name, self.manifest, dict(producer_run_id='synthetic-test',
            checkpoint_sha256=self.checkpoint_hash, export_config_sha256=fingerprint(self.config)),
            self.config['decoder'])

    def append(self, writer, iid, candidates):
        writer.append(iid, candidates, dict(class_logits_shape=[100, 9],
            mask_logits_original_shape=[100, 4, 6], all_logits_finite=True, candidate_count=len(candidates)))

    def complete(self):
        w = self.writer()
        # Sparse real query indexes, tied scores, an exact float32 .7 boundary.
        self.append(w, 17, [self.candidate(4, .9), self.candidate(73, .9),
                           self.candidate(98, float(np.float32(.7))), self.candidate(2, .2)])
        self.append(w, 23, [])
        w.finish()
        return w

    def export(self, w, name='out', threshold=.7):
        return export_cached(w.root, self.root/name, threshold=threshold, candidate_set_id=name,
                             expected_checkpoint_sha256=self.checkpoint_hash)

    def test_sparse_query_ids_ties_threshold_and_empty_image(self):
        w = self.complete()
        for t in (.3, .5, .7):
            p = self.export(w, str(t), t)
            self.assertEqual([c['decoder_query_id'] for c in p['images'][0]['candidates']], [4, 73, 98])
            self.assertEqual(p['images'][1]['status'], 'no_candidates')
            self.assertTrue(p['illustrative_only'])
            self.assertIsNone(p['provenance']['annotation_manifest_sha256'])
            self.assertEqual(p['provenance']['threshold_status'], 'trial_unselected')
        self.assertEqual(read_json(w.root/'records/17.json')['candidates'][-1]['decoder_query_id'], 2)

    def test_synthetic_decoder_and_cache_filter_agree(self):
        # Existing decoder only; no model construction/weights/forward pass.
        import torch
        from fashion_grounding.segmentation.predict import decode_candidates
        logits = torch.full((100, 9), -4.)
        logits[:, 8] = 4.
        logits[[4, 73, 98], 1] = torch.tensor([9., 7., 5.])
        masks = torch.full((100, 4, 6), -2.)
        masks[[4, 73, 98], 1:3, 2:5] = 2.
        w = self.writer()
        self.append(w, 17, decode_candidates(logits, masks, 17, 0))
        self.append(w, 23, [])
        w.finish()
        for t in (.3, .5, .7):
            observed = self.export(w, f'decoder-{t}', t)['images'][0]['candidates']
            expected = decode_candidates(logits, masks, 17, t)
            self.assertEqual([(c['decoder_query_id'], c['score']) for c in observed],
                             [(c.decoder_query_id, c.score) for c in expected])

    def test_streaming_orchestration_uses_only_pixels_and_unfiltered_settings(self):
        w = self.writer()
        calls = []
        def fake(pixels, settings, diag):
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertEqual(pixels.shape, (4, 6, 3))
            self.assertEqual(settings['score_threshold'], 0)
            self.assertEqual(settings['max_candidates'], 100)
            self.assertFalse(any('annotation' in k or 'target' in k for k in settings))
            calls.append(settings['image_id'])
            diag.update(class_logits_shape=[100, 9], mask_logits_original_shape=[100, 4, 6],
                        all_logits_finite=True, candidate_count=0)
            return []
        cache_images(w, fake, self.config, deadline=time.monotonic()+10)
        self.assertEqual(calls, [17, 23])
        self.assertEqual(len(read_json(w.root/'cache.json')['records']), 2)

    def test_failure_incomplete_and_deadline_cannot_export(self):
        w = self.writer()
        self.append(w, 17, [])
        with self.assertRaisesRegex(ValueError, 'Incomplete cache'):
            w.finish()
        self.assertFalse((w.root/'cache.json').exists())
        w2 = self.writer('failed')
        def fail(*args):
            raise RuntimeError('synthetic failure')
        with self.assertRaises(RuntimeError):
            cache_images(w2, fail, self.config, deadline=time.monotonic()+10)
        self.assertEqual(read_json(w2.root/'run_metadata.json')['status'], 'failed')
        self.assertFalse((w2.root/'cache.json').exists())
        w3 = self.writer('expired')
        with self.assertRaisesRegex(ValueError, 'time budget'):
            cache_images(w3, fail, self.config, deadline=0)
        self.assertFalse((w3.root/'cache.json').exists())

    def test_cache_and_mask_tampering_and_wrong_checkpoint_rejected(self):
        w = self.complete()
        with self.assertRaisesRegex(ValueError, 'Wrong checkpoint'):
            export_cached(w.root, self.root/'wrong', threshold=.5, candidate_set_id='wrong',
                          expected_checkpoint_sha256='f'*64)
        (w.root/'masks/17-4.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Mask hash mismatch'):
            self.export(w, 'bad-mask')
        self.assertFalse((self.root/'bad-mask/candidates_predicted.json').exists())
        (w.root/'cache.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Cache hash mismatch'):
            self.export(w, 'bad-root')

    def test_duplicate_query_bad_geometry_and_changed_query_count_rejected(self):
        for name, cs, message in [('duplicate', [self.candidate(4,.9)]*2, 'duplicate'),
                                  ('geometry', [self.candidate(4,.9)], 'Mask/box')]:
            w = self.writer(name)
            if name == 'geometry':
                cs[0].bbox_xyxy = (0, 0, 1, 1)
            with self.assertRaisesRegex(ValueError, message):
                self.append(w, 17, cs)
        w = self.writer('queries')
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            w.append(17, [], dict(class_logits_shape=[101, 9], mask_logits_original_shape=[101, 4, 6],
                                 all_logits_finite=True, candidate_count=0))

    def test_input_boundary_and_paths(self):
        manifest = read_json(self.manifest)
        manifest['illustrative_only'] = False
        write_json(self.manifest, manifest)
        c = copy.deepcopy(self.config)
        c['inputs']['dev'] = dict(image_manifest_sha256=sha256(self.manifest), images=2)
        # No annotations or queries exist, yet image-only preflight succeeds.
        self.assertEqual(len(input_manifest(self.source, c, 'dev')[1]), 2)
        manifest['images'][0]['split'] = 'test'
        write_json(self.manifest, manifest)
        c['inputs']['dev']['image_manifest_sha256'] = sha256(self.manifest)
        with self.assertRaisesRegex(ValueError, 'train/dev'):
            input_manifest(self.source, c, 'dev')
        manifest['images'][0]['split'] = 'dev'
        manifest['images'][0]['file_name'] = '../outside.png'
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, 'unsafe path'):
            self.append(self.writer(), 17, [])

    def test_metadata_guard_without_loading_checkpoint(self):
        c = copy.deepcopy(self.config)
        state = dict(completed_updates=3000, scheduler=dict(completed_updates=3000),
            config=dict(purpose='s0_training'), git_commit=c['training_git_commit'], data=c['training_data'])
        state['config_sha256'] = c['training_config_sha256'] = fingerprint(state['config'])
        validate_checkpoint_metadata(state, c)
        state['completed_updates'] = 5000
        with self.assertRaisesRegex(ValueError, 'checkpoint update'):
            validate_checkpoint_metadata(state, c)
        validate_config(self.config)

    def test_output_is_immutable(self):
        w = self.complete()
        self.export(w)
        with self.assertRaises(FileExistsError):
            self.export(w)
        with self.assertRaises(FileExistsError):
            self.writer()

    def test_offline_cli_does_not_import_torch_or_model(self):
        w = self.complete()
        c = copy.deepcopy(self.config)
        c['inputs']['dev'] = dict(image_manifest_sha256=sha256(self.manifest), images=2)
        cache = read_json(w.root/'cache.json')
        cache['provenance']['export_config_sha256'] = fingerprint(c)
        write_json(w.root/'cache.json', cache)
        (w.root/'cache.sha256').write_text(sha256(w.root/'cache.json')+'\n')
        write_json(self.root/'config.json', c)
        code = '''import sys
from fashion_grounding.segmentation.export_candidates import main
main()
assert 'torch' not in sys.modules
assert 'fashion_grounding.segmentation.model' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code, 'export', '--config', str(self.root/'config.json'),
            '--cache', str(w.root), '--output', str(self.root/'cli'), '--threshold', '0.7',
            '--candidate-set-id', 'synthetic-cli'], check=True)
        self.assertEqual(read_json(self.root/'cli/run_metadata.json')['model_inference_calls'], 0)

    @unittest.skipUnless(os.environ.get('GROUNDING_INPUTS_PATH'), 'Pinned consumer supplied by test runner')
    def test_wyz_pinned_consumer_reads_export(self):
        w = self.complete()
        self.export(w)
        spec = importlib.util.spec_from_file_location('pinned_grounding_inputs', os.environ['GROUNDING_INPUTS_PATH'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pack = module.CandidatePack(self.root/'out/candidates_predicted.json', allow_illustrative=True)
        image, candidates = pack.load(17)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(module.crop_regions(image, candidates)), 3)
        self.assertEqual(len(pack.load(23)[1]), 0)


if __name__ == '__main__':
    unittest.main()
