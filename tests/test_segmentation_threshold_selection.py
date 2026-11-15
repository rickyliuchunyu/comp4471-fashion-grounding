"""New offline selection behavior, using tiny synthetic masks; no model/smoke."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from fashion_grounding.data.schema import encode_mask, read_json, sha256, write_json
from fashion_grounding.evaluation import matching
from fashion_grounding.segmentation.select_threshold import (
    choose_threshold, image_coverage, load_frozen_dev, run)


class ThresholdSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.left = np.zeros((4, 4), dtype=bool)
        self.left[:, :2] = True
        self.right = ~self.left
        self.images = [dict(image_id=i, width=4, height=4, sha256=str(i)*64,
                            group_id=str(i), split='dev', file_name=f'{i}.png') for i in (1, 2, 3)]
        write_json(self.root/'images.json', dict(schema_version=3, illustrative_only=False, images=self.images))
        anns = [dict(id=aid, image_id=iid, category_id=1, iscrowd=0, area=8,
                     segmentation=encode_mask(mask))
                for aid, iid, mask in ((11, 1, self.left), (12, 1, self.right),
                                      (21, 2, self.left), (31, 3, self.left))]
        self.anns = anns
        write_json(self.root/'annotations.json', dict(schema_version=3, illustrative_only=False,
            images=[dict(id=i, width=4, height=4) for i in (1, 2, 3)], annotations=anns))
        self.queries = [dict(schema_version=3, illustrative_only=False,
            query_id=f'{aid}-{lang}', semantic_query_id=str(aid), image_id=iid,
            target_instance_id=aid, target_category_id=1, split='dev', query='synthetic query',
            language=lang, review_status='accepted', annotator='fixture-author', reviewer='fixture-reviewer')
            for aid, iid, lang in ((11, 1, 'zh'), (11, 1, 'en'), (12, 1, 'en'), (21, 2, 'en'))]
        self.save_queries()
        (self.root/'review.txt').write_text('SYNTHETIC TEST ONLY: not human review or project evidence.')
        self.freeze = dict(schema_version=1, purpose='accepted_language_dev_freeze',
            formal_freeze=True, split='dev', confirmed_by='fixture-human', confirmed_at='fixture-only',
            queries=self.ref('queries.jsonl'), image_manifest=self.ref('images.json'),
            annotations=self.ref('annotations.json'), review_records=[self.ref('review.txt')],
            scope=dict(kind='accepted_dev_subset', description='Synthetic 2-of-3 image fixture',
                       query_count=4, query_image_ids=[1, 2]))
        self.config = dict(schema_version=1, purpose='offline_candidate_threshold_selection',
            checkpoint_sha256='a'*64, source_cache_sha256='b'*64,
            image_manifest_sha256=sha256(self.root/'images.json'),
            annotation_sha256=sha256(self.root/'annotations.json'), matcher_sha256=sha256(matching.__file__),
            packages=[])
        self.candidates = []
        for cid, category, score, mask in [('wrong-class', 2, .9, self.left),
                ('left', 1, .6, self.left), ('duplicate-left', 1, .55, self.left),
                ('right', 1, .4, self.right)]:
            write_json(self.root/f'{cid}.json', encode_mask(mask))
            self.candidates.append(dict(candidate_id=cid, category_id=category, score=score,
                mask_reference=f'{cid}.json', mask_sha256=sha256(self.root/f'{cid}.json')))
        for threshold in (.3, .5, .7):
            package = dict(schema_version=3, illustrative_only=False, candidate_source='predicted',
                candidate_set_id=f'fixture-{threshold}', image_manifest=self.ref('images.json'),
                provenance=dict(checkpoint_sha256='a'*64, source_cache_sha256='b'*64,
                    source_image_manifest_sha256=self.config['image_manifest_sha256'], score_threshold=threshold),
                images=[dict(image_id=1, image_size_wh=[4, 4], status='ok',
                             candidates=[c for c in self.candidates if c['score'] >= threshold]),
                        dict(image_id=2, image_size_wh=[4, 4], status='no_candidates', candidates=[]),
                        # No queries on image 3: its candidates MUST NOT break selection ties/counts.
                        dict(image_id=3, image_size_wh=[4, 4], status='ok',
                             candidates=[dict(self.candidates[0], candidate_id=f'unqueried-{i}') for i in range(10)])])
            name = f'pack-{threshold}.json'
            write_json(self.root/name, package)
            self.config['packages'].append(dict(threshold=threshold, **self.ref(name)))
        self.save_inputs()

    def ref(self, name):
        return dict(path=name, sha256=sha256(self.root/name))

    def save_queries(self):
        (self.root/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in self.queries))

    def save_inputs(self):
        write_json(self.root/'freeze.json', self.freeze)
        write_json(self.root/'config.json', self.config)

    def execute(self, name='out'):
        return run(self.root/'config.json', self.root/'freeze.json', self.root/name)

    def test_real_entry_coverage_geometry_empty_denominator_and_no_pixels(self):
        # No RGB files exist. Only persisted masks/metadata are needed.
        result = self.execute()
        self.assertEqual(result['selected_threshold'], .3)
        self.assertEqual(result['selection_reason'], 'most_covered_queries')
        summaries = result['comparison']
        self.assertEqual([r['covered_queries'] for r in summaries], [3, 2, 0])
        self.assertEqual([r['geometric_recalled_queries'] for r in summaries], [3, 2, 2])
        self.assertEqual([r['candidate_count_on_query_images'] for r in summaries], [4, 3, 1])
        self.assertTrue(all(r['queries'] == 4 and r['images'] == 2 and r['zero_candidate_images'] == 1 for r in summaries))
        self.assertEqual(summaries[0]['matched_candidate_coverage'], .75)
        rows = read_json(self.root/'out/coverage-0.3.json')
        self.assertEqual(rows['association']['1'], {'left': 11, 'right': 12})
        self.assertFalse(rows['rows'][-1]['covered'])
        self.assertEqual(read_json(self.root/'out/run_metadata.json')['status'], 'completed')
        self.assertEqual(len((self.root/'out/comparison.csv').read_text().splitlines()), 4)
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            self.execute()

    def test_all_gt_participate_in_association_even_without_query(self):
        # Equal-IoU unqueried GT 10 precedes queried GT 11. Query-only matching would falsely cover 11.
        anns = [dict(self.anns[0], id=10), self.anns[0]]
        q = self.queries[:1]
        c = self.candidates[1]
        rows, association = image_coverage(q, anns, dict(candidates=[c], status='ok'),
            {'left': self.left}, {10: self.left, 11: self.left})
        self.assertEqual(association, {'left': 10})
        self.assertFalse(rows[0]['covered'])
        self.assertTrue(rows[0]['geometric_recalled'])

    def test_exact_half_iou_score_and_id_ties_crowd(self):
        full = np.ones((4, 4), dtype=bool)
        a = self.anns[0]
        anns = [dict(a, id=1, iscrowd=1), dict(a, id=10), a]
        cs = [dict(candidate_id=cid, score=.8, category_id=1) for cid in ('b', 'a')]
        rows, association = image_coverage(self.queries[:1], anns, dict(candidates=cs, status='ok'),
            {'a': full, 'b': full}, {1: self.left, 10: self.left, 11: self.left})
        self.assertEqual(association, {'a': 10, 'b': 11})
        self.assertTrue(rows[0]['covered'])

    def test_selection_count_and_high_threshold_ties(self):
        def trial(t, counts):
            return dict(threshold=t, rows=[dict(query_id=str(i), image_id=i//2, candidate_count=counts[i//2],
                covered=False, geometric_recalled=True) for i in range(4)])
        # All zero matched coverage is still a valid nonempty denominator.
        rows = [trial(.3, [4, 0]), trial(.5, [2, 0]), trial(.7, [3, 0])]
        self.assertEqual(choose_threshold(rows), (.5, 'fewest_candidates_on_unique_query_images'))
        self.assertEqual(rows[1]['summary']['candidate_count_on_query_images'], 2)
        rows[2] = trial(.7, [2, 0])
        self.assertEqual(choose_threshold(rows), (.7, 'higher_threshold'))
        rows[1]['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'denominators'):
            choose_threshold(rows)

    def test_freeze_scope_evidence_and_hash_gates(self):
        for mutate in (
            lambda f: f.update(formal_freeze=False),
            lambda f: f.update(split='test'),
            lambda f: f.update(review_records=[]),
            lambda f: f.update(confirmed_by=None),
            lambda f: f['scope'].update(query_count=3),
            lambda f: f['scope'].update(query_image_ids=[1]),
            lambda f: f['scope'].update(kind='full_dev'),
            lambda f: f['queries'].update(sha256='0'*64),
        ):
            freeze = copy.deepcopy(self.freeze)
            mutate(freeze)
            write_json(self.root/'bad-freeze.json', freeze)
            with self.subTest(freeze=freeze), self.assertRaises(ValueError):
                load_frozen_dev(self.root/'bad-freeze.json', self.config)

    def test_query_gates_do_not_open_candidate_files(self):
        original = copy.deepcopy(self.queries)
        for problem in ('empty', 'unreviewed', 'train', 'duplicate', 'self_review', 'wrong_target'):
            self.queries = copy.deepcopy(original)
            if problem == 'empty': self.queries = []
            if problem == 'unreviewed': self.queries[0]['review_status'] = 'needs_revision'
            if problem == 'train': self.queries[0]['split'] = 'train'
            if problem == 'duplicate': self.queries.append(self.queries[0])
            if problem == 'self_review': self.queries[0]['reviewer'] = 'fixture-author'
            if problem == 'wrong_target': self.queries[0]['target_instance_id'] = 21
            self.save_queries()
            self.freeze['queries'] = self.ref('queries.jsonl')
            self.save_inputs()
            # Invalid queries must fail with ValueError, before missing candidate file lookup.
            for p in self.config['packages']: (self.root/p['path']).unlink(missing_ok=True)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                self.execute(problem)
            self.assertFalse((self.root/problem).exists())

    def test_wrong_candidate_provenance_or_inventory_rejected(self):
        spec = self.config['packages'][0]
        original = read_json(self.root/spec['path'])
        for field in ('checkpoint', 'cache', 'threshold', 'missing_image', 'pixel_identity'):
            package = copy.deepcopy(original)
            if field == 'checkpoint': package['provenance']['checkpoint_sha256'] = 'c'*64
            if field == 'cache': package['provenance']['source_cache_sha256'] = 'c'*64
            if field == 'threshold': package['provenance']['score_threshold'] = .5
            if field == 'missing_image': package['images'].pop()
            if field == 'pixel_identity':
                manifest = read_json(self.root/'images.json')
                manifest['images'][0]['sha256'] = '0'*64
                write_json(self.root/'wrong-images.json', manifest)
                package['image_manifest'] = self.ref('wrong-images.json')
            write_json(self.root/spec['path'], package)
            spec['sha256'] = sha256(self.root/spec['path'])
            self.save_inputs()
            with self.subTest(field=field), self.assertRaises(ValueError): self.execute(field)
            self.assertFalse((self.root/field/'selection.json').exists())

    def test_mask_corruption_leaves_failed_run_without_selection(self):
        (self.root/'wrong-class.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'mask hash'):
            self.execute()
        self.assertFalse((self.root/'out/selection.json').exists())
        self.assertEqual(read_json(self.root/'out/run_metadata.json')['status'], 'failed')

    def test_cli_is_model_free_and_keeps_inputs_immutable(self):
        before = {p: sha256(p) for p in self.root.iterdir() if p.is_file()}
        code = '''
import importlib.abc, runpy, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'detectron2'):
            raise RuntimeError('Forbidden model import: '+fullname)
sys.meta_path.insert(0, BlockModels())
runpy.run_module('fashion_grounding.segmentation.select_threshold', run_name='__main__')
'''
        p = subprocess.run([sys.executable, '-c', code, '--config', str(self.root/'config.json'),
            '--freeze', str(self.root/'freeze.json'), '--output', str(self.root/'cli'),
            '--budget-seconds', '10'], capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('Selected threshold: 0.3', p.stdout)
        self.assertEqual(before, {p: sha256(p) for p in before})

    def test_budget_and_matcher_pin_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'time budget'):
            run(self.root/'config.json', self.root/'freeze.json', self.root/'budget', budget_seconds=1e-12)
        self.config['matcher_sha256'] = 'f'*64
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'matcher version'):
            self.execute()


if __name__ == '__main__':
    unittest.main()
