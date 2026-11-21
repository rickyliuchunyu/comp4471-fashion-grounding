"""New same-budget transition gates; no production weight/model/data loading."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from fashion_grounding.segmentation.resume_repair import validate_repair, load_repaired_checkpoint, REPAIR_FILES
from fashion_grounding.segmentation.training_state import read_config, config_sha256, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class ResumeRepairTests(unittest.TestCase):
    def setUp(self):
        self.old = read_config(ROOT/'configs/runs/segmentation/s0_resume_20000.json')
        self.new = {**self.old, 'amp_recovery': dict(max_retries_per_update=4,
                    max_overflow_events=16, min_loss_scale=0.0001)}
        self.state = dict(config=self.old, config_sha256=config_sha256(self.old),
                          git_commit='parent', completed_updates=18000)

    def policy(self, config=None):
        config = self.new if config is None else config
        return dict(schema_version=1, transition_type='amp_recovery_same_budget_v1',
                    parent_checkpoint_sha256='a'*64, parent_git_commit='parent',
                    parent_config_sha256=config_sha256(self.old), parent_completed_updates=18000,
                    target_config_sha256=config_sha256(config), parent_source_tree='old',
                    target_source_tree='new', changed_source_files=sorted(REPAIR_FILES), reason='AMP fix',
                    allowed_changes={k:dict(before=self.old.get(k), after=config.get(k))
                        for k in self.old.keys() | config.keys() if self.old.get(k) != config.get(k)})

    def validate(self, config=None, policy=None, state=None, files=None):
        return validate_repair(state or self.state, config or self.new, policy or self.policy(),
            'a'*64, 'current', 'old', 'new', files or sorted(REPAIR_FILES))

    def test_same_budget_transition_does_not_mutate_state(self):
        before = copy.deepcopy(self.state)
        result = self.validate()
        self.assertEqual(before, self.state)
        self.assertFalse(result['update_budget_extended'])
        self.assertEqual(result['restored_state_overrides'], {})

    def test_recipe_budget_and_unknown_source_changes_rejected_even_if_policy_matches(self):
        for key, value in [('max_updates', 21000), ('learning_rate', 0.01), ('seed', 2),
                           ('image_size', 640), ('amp_initial_scale', 4), ('annotation_sha256', 'f'*64)]:
            new = {**self.new, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config=new, policy=self.policy(new))
        files = sorted(REPAIR_FILES | {'src/fashion_grounding/segmentation/evaluate.py'})
        with self.assertRaisesRegex(ValueError, 'Unreviewed source'):
            self.validate(policy={**self.policy(), 'changed_source_files': files}, files=files)

    def test_all_pinned_identities_are_enforced(self):
        for key in ('parent_checkpoint_sha256', 'parent_git_commit', 'parent_config_sha256',
                    'target_config_sha256', 'parent_source_tree', 'target_source_tree',
                    'parent_completed_updates', 'transition_type'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(policy={**self.policy(), key: 'wrong'})
        with self.assertRaises(ValueError): self.validate(policy={**self.policy(), 'allowed_changes': {}})
        with self.assertRaises(ValueError): self.validate(state={**self.state, 'config_sha256': 'bad'})

    def test_loader_checks_actual_checkpoint_and_git_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            git('init'); git('config', 'user.email', 'fixture@example.invalid'); git('config', 'user.name', 'Fixture')
            p = root/'src/fashion_grounding/segmentation/train.py'
            p.parent.mkdir(parents=True); p.write_text('old\n'); git('add', '.')
            git('-c', 'commit.gpgsign=false', 'commit', '-m', 'old')
            parent = git('rev-parse', 'HEAD'); old_tree = git('rev-parse', 'HEAD:src/fashion_grounding')
            p.write_text('repaired\n'); git('add', '.')
            git('-c', 'commit.gpgsign=false', 'commit', '-m', 'repair')
            current = git('rev-parse', 'HEAD'); new_tree = git('rev-parse', 'HEAD:src/fashion_grounding')
            state = {**self.state, 'git_commit': parent, 'data': {'hash': 'data'},
                     'scheduler': {'completed_updates': 18000}, 'scaler': {'scale': 512.}}
            checkpoint = root/'state.pth'; digest = save_checkpoint(checkpoint, state)
            policy = {**self.policy(), 'parent_checkpoint_sha256': digest, 'parent_git_commit': parent,
                      'parent_source_tree': old_tree, 'target_source_tree': new_tree,
                      'changed_source_files': ['src/fashion_grounding/segmentation/train.py']}
            path = root/'repair.json'; path.write_text(json.dumps(policy))
            loaded, record = load_repaired_checkpoint(checkpoint, self.new, state['data'], path, current, root)
            self.assertEqual(loaded, state); self.assertEqual(loaded['scaler']['scale'], 512)
            self.assertFalse(record['update_budget_extended'])
            with self.assertRaisesRegex(ValueError, 'dataset'):
                load_repaired_checkpoint(checkpoint, self.new, {'hash': 'wrong'}, path, current, root)
            policy['parent_checkpoint_sha256'] = 'b'*64; path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                load_repaired_checkpoint(checkpoint, self.new, state['data'], path, current, root)


if __name__ == '__main__': unittest.main()
