"""Only new incremental acceptance invariants; no data/model smoke."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('accept_expansion', Path(__file__).resolve().parents[1]/'scripts/accept_segmentation_expansion.py')
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


class ExpansionAcceptanceTests(unittest.TestCase):
    def test_exact_old_subset_and_delta(self):
        old = [{'id': 1, 'mask': 'original'}]
        new = {'id': 2, 'mask': 'added'}
        self.assertEqual(accept.check_subsets(old, old+[new], [1], [2], 'id'), [new])
        changed = [{'id': 1, 'mask': 'changed'}, new]
        with self.assertRaisesRegex(ValueError, 'old record changed'):
            accept.check_subsets(old, changed, [1], [2], 'id')

    def test_duplicate_and_incomplete_memberships_rejected(self):
        for merged, old_ids, added_ids in [([{'id': 1}, {'id': 2}, {'id': 2}], [1], [2]),
                                           ([{'id': 1}, {'id': 2}], [1], []),
                                           ([{'id': 1}, {'id': 2}], [1], [1, 2])]:
            with self.assertRaises(ValueError):
                accept.check_subsets([{'id': 1}], merged, old_ids, added_ids, 'id')

    def test_source_id_namespace_and_isolation(self):
        added = dict(image_id=8, source_dataset='fashionpedia', source_split='official_train2020', source_image_id='7', group_id='train-7', sha256='a')
        dev = dict(image_id=1000000008, source_dataset='fashionpedia', source_split='official_val2020', source_image_id='7', group_id='val-7', sha256='b')
        self.assertEqual(accept.check_isolation([added], {'dev': [dev]})['dev']['source_identity'], 0)
        for field in ('image_id', 'group_id', 'sha256'):
            bad = {**dev, field: added[field]}
            with self.assertRaisesRegex(ValueError, 'cross-partition'):
                accept.check_isolation([added], {'dev': [bad]})
        with self.assertRaisesRegex(ValueError, 'cross-partition'):
            accept.check_isolation([added], {'train': [{**dev, 'source_split': 'official_train2020'}]})

    def test_duplicate_added_groups_rejected(self):
        row = dict(image_id=1, source_dataset='fashionpedia', source_split='official_train2020', source_image_id='0', group_id='same', sha256='a')
        second = {**row, 'image_id': 2, 'source_image_id': '1', 'sha256': 'b'}
        with self.assertRaisesRegex(ValueError, 'duplicate added'):
            accept.check_isolation([row, second], {})


if __name__ == '__main__':
    unittest.main()
