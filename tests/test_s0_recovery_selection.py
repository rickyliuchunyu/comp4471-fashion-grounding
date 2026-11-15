"""New cross-repair identity audit only; reuse the established AP evaluator."""
import copy
import json
from pathlib import Path
import unittest

from scripts.select_s0_recovery import select_reports


class RecoverySelectionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.prior = json.loads((root/'docs/handoffs/segmentation/evidence/resume5000_20261113/update_003000/evaluation_report.json').read_text())
        self.reports, self.expected = [], {}
        for u in range(6000, 20001, 2000):
            identity = dict(checkpoint_sha256=str(u), checkpoint_path=f'checkpoint_{u}.pth',
                            training_git_commit='repaired' if u == 20000 else 'parent',
                            training_config_sha256='new' if u == 20000 else 'old')
            self.expected[u] = identity
            self.reports.append({**copy.deepcopy(self.prior), **identity, 'completed_updates': u,
                                 'evaluation_git_commit': 'eval', 'evaluation_config_sha256': 'cfg'})

    def select(self):
        return select_reports(self.reports, self.prior, self.expected, 'eval', 'cfg')

    def test_tie_retains_prior_and_new_best_can_win(self):
        self.assertTrue(self.select()['old_best_retained'])
        self.reports[-1]['metrics']['AP'] += 1
        self.assertEqual(self.select()['selected']['completed_updates'], 20000)
        self.reports[1]['metrics']['AP'] += 1
        self.assertEqual(self.select()['selected']['completed_updates'], 8000)

    def test_incomplete_duplicate_and_invalid_score_rejected(self):
        for kind in ('missing', 'duplicate', 'incomplete', 'nan'):
            self.setUp()
            if kind == 'missing': self.reports.pop()
            elif kind == 'duplicate': self.reports[-1] = self.reports[0]
            elif kind == 'incomplete': self.reports[-1]['complete'] = False
            else: self.reports[-1]['metrics']['AP'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.select()

    def test_source_config_checkpoint_and_denominator_changes_rejected(self):
        for key, value in [('training_git_commit', 'unknown'), ('training_config_sha256', 'wrong'),
                           ('checkpoint_sha256', 'wrong'), ('checkpoint_path', '/wrong'),
                           ('evaluation_git_commit', 'old'), ('evaluation_config_sha256', 'wrong'),
                           ('evaluated_images', 249), ('optimizer_updates', 1), ('queries_read', True),
                           ('evaluation_data', {}), ('training_data', {}), ('overlap', {})]:
            self.setUp(); self.reports[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.select()


if __name__ == '__main__': unittest.main()
