"""Only the new selected-checkpoint config gate; no model or inference."""
import copy
import unittest
from pathlib import Path

from fashion_grounding.data.schema import read_json
from fashion_grounding.segmentation.export_candidates import validate_config


class SelectedCandidateConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]/'configs/runs/segmentation'
        self.config = read_json(self.root/'candidate_u18000_export.json')

    def test_selected_and_legacy_configs_keep_exact_decoder(self):
        validate_config(self.config)
        legacy = read_json(self.root/'candidate_u3000_export.json')
        validate_config(legacy)
        self.assertEqual(self.config['decoder'], legacy['decoder'])
        changed = copy.deepcopy(self.config)
        changed['decoder']['score_threshold'] = .3
        with self.assertRaisesRegex(ValueError, 'decoder configuration'):
            validate_config(changed)

    def test_invalid_updates_and_legacy_mislabel_are_rejected(self):
        for updates in (0, -1, True, 18000.0, '18000'):
            with self.subTest(updates=updates):
                changed = dict(self.config, completed_updates=updates)
                with self.assertRaisesRegex(ValueError, 'selected-checkpoint'):
                    validate_config(changed)
        for purpose in ('u3000_candidate_export', 's0_training'):
            with self.assertRaisesRegex(ValueError, 'selected-checkpoint'):
                validate_config(dict(self.config, purpose=purpose))


if __name__ == '__main__':
    unittest.main()
