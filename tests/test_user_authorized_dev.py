"""Only the new explicit dev authorization path; reuse the existing tiny fixture."""
import copy
import json
import unittest

import test_segmentation_threshold_selection as fixtures
from fashion_grounding.data.schema import read_json, sha256, write_json
from fashion_grounding.segmentation.select_threshold import load_frozen_dev, run


class UserAuthorizedDevTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.ThresholdSelectionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for q in self.f.queries:
            q.update(review_status='needs_revision', reviewer=None)
        self.f.save_queries()
        self.authorization = dict(schema_version=1, authority='current_user',
            authorized_use='offline_candidate_threshold_selection',
            user_statement='SYNTHETIC authorization fixture, not actual project evidence',
            confirmed_by='fixture-user', confirmed_at='fixture-date',
            queries_sha256=sha256(self.f.root/'queries.jsonl'))
        write_json(self.f.root/'authorization.json', self.authorization)
        self.f.freeze.update(purpose='user_authorized_language_dev', formal_freeze=False,
            development_freeze=True, confirmed_by='fixture-user', confirmed_at='fixture-date',
            queries=self.f.ref('queries.jsonl'), review_records=[],
            authorization_record=self.f.ref('authorization.json'))
        self.f.freeze['scope']['kind'] = 'user_authorized_dev_subset'
        self.f.save_inputs()

    def test_opt_in_and_bound_authorization_required(self):
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            load_frozen_dev(self.f.root/'freeze.json', self.f.config)
        bad = dict(self.authorization, queries_sha256='0'*64)
        write_json(self.f.root/'authorization.json', bad)
        self.f.freeze['authorization_record'] = self.f.ref('authorization.json')
        self.f.save_inputs()
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            load_frozen_dev(self.f.root/'freeze.json', self.f.config,
                            allow_user_authorized_dev=True)

    def test_dev_and_target_and_default_accepted_guards_remain(self):
        original = copy.deepcopy(self.f.queries)
        for field, value in [('split', 'train'), ('target_category_id', 8)]:
            self.f.queries = copy.deepcopy(original)
            self.f.queries[0][field] = value
            self.f.save_queries()
            self.authorization['queries_sha256'] = sha256(self.f.root/'queries.jsonl')
            write_json(self.f.root/'authorization.json', self.authorization)
            self.f.freeze.update(queries=self.f.ref('queries.jsonl'),
                                 authorization_record=self.f.ref('authorization.json'))
            self.f.save_inputs()
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_frozen_dev(self.f.root/'freeze.json', self.f.config,
                                allow_user_authorized_dev=True)
        self.f.queries = original
        self.f.save_queries()
        self.f.freeze.update(purpose='accepted_language_dev_freeze', formal_freeze=True,
            queries=self.f.ref('queries.jsonl'), review_records=[self.f.ref('review.txt')])
        self.f.freeze['scope']['kind'] = 'accepted_dev_subset'
        self.f.save_inputs()
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            load_frozen_dev(self.f.root/'freeze.json', self.f.config,
                            allow_user_authorized_dev=True)

    def test_selection_keeps_authorization_basis_and_original_query_bytes(self):
        path = self.f.root/'queries.jsonl'
        before = path.read_bytes()
        result = run(self.f.root/'config.json', self.f.root/'freeze.json',
                     self.f.root/'authorized-out', allow_user_authorized_dev=True)
        self.assertEqual(result['selected_threshold'], .3)
        self.assertEqual(result['status'], 'selected_for_user_authorized_development')
        self.assertEqual(result['source_review_status_counts'], {'needs_revision': 4})
        self.assertEqual(path.read_bytes(), before)
        for name in ['run_metadata.json', 'coverage-0.3.json', 'coverage-0.5.json',
                     'coverage-0.7.json', 'selection.json']:
            report = read_json(self.f.root/'authorized-out'/name)
            self.assertEqual(report['evaluation_basis'], 'user_authorized_development')
        self.assertTrue(all(json.loads(line)['reviewer'] is None
                            for line in before.decode().splitlines()))


if __name__ == '__main__':
    unittest.main()
