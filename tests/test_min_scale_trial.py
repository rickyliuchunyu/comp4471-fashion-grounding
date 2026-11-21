"""New transition, offline comparison and deadline gates only; no model/smoke."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fashion_grounding.segmentation.resume_recipe import validate_recipe, load_recipe_checkpoint
from fashion_grounding.segmentation.training_state import config_sha256
from run_min_scale_trial import build_commands, run_bounded, worker, arm_worker_deadline
from select_min_scale_trial import compare_reports


def read(path):
    return json.loads((ROOT/path).read_text())


class RecipeTests(unittest.TestCase):
    def setUp(self):
        base = 'configs/runs/segmentation/'
        self.old = read(base+'s0_resume_20000.json')
        self.a = read(base+'s0_recovery_20000.json')
        self.b = read(base+'min_scale_05_20000.json')
        self.repair = read(base+'resume_repair_18000_to_20000.json')
        self.policy = read(base+'resume_recipe_18000_to_20000.json')
        self.state = dict(config=self.old, config_sha256=config_sha256(self.old),
                          git_commit=self.policy['parent_git_commit'], completed_updates=18000,
                          **{k: {'sentinel': [1, 2, 3]} for k in
                             ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'sampler')})

    def validate(self, config=None, policy=None, state=None, files=None):
        p = self.policy if policy is None else policy
        return validate_recipe(self.state if state is None else state, self.b if config is None else config,
            self.a, self.repair, p, self.policy['parent_checkpoint_sha256'], 'current',
            self.policy['parent_source_tree'], self.policy['target_source_tree'],
            self.policy['changed_source_files'] if files is None else files)

    def policy_for(self, config):
        return {**self.policy, 'target_config_sha256': config_sha256(config),
                'control_to_target_changes': {k: dict(before=self.a.get(k), after=config.get(k))
                    for k in self.a.keys() | config.keys() if self.a.get(k) != config.get(k)}}

    def test_only_factor_changes_and_six_states_are_not_mutated(self):
        before = copy.deepcopy(self.state)
        audit = self.validate()
        self.assertEqual(before, self.state)
        self.assertEqual(audit['restored_state_overrides'], {})
        self.assertFalse(audit['update_budget_extended'])
        self.assertEqual(audit['control_to_target_changes']['min_scale'], {'before': 0.1, 'after': 0.5})

    def test_other_factors_rejected_even_with_matching_policy_hash(self):
        for key, value in [('learning_rate', 0.001), ('lr_milestones', [19000]),
            ('seed', 43), ('image_size', 1024), ('max_scale', 1.5), ('random_flip', 'none'),
            ('amp_recovery', None), ('microbatch_images', 4), ('annotation_sha256', 'b'*64),
            ('image_manifest_sha256', 'c'*64), ('gradient_clip_norm', 0.1), ('new_loss', True),
            ('min_scale', 0.4), ('max_updates', 21000), ('budget_seconds', 2101),
            ('checkpoint_updates', [19000, 20000])]:
            config = {**self.b, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config, self.policy_for(config))

    def test_checkpoint_source_and_config_identities_fail_closed(self):
        for key in ('transition_type', 'parent_checkpoint_sha256', 'parent_git_commit',
                    'parent_config_sha256', 'parent_completed_updates', 'target_config_sha256',
                    'control_config_sha256', 'parent_source_tree', 'target_source_tree'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(policy={**self.policy, key: 'wrong'})
        with self.assertRaises(ValueError):
            self.validate(state={**self.state, 'config_sha256': 'wrong'})
        files = self.policy['changed_source_files']+['src/fashion_grounding/segmentation/evaluate.py']
        with self.assertRaises(ValueError):
            self.validate(policy={**self.policy, 'changed_source_files': sorted(files)}, files=files)

    def test_new_loader_preserves_old_loader_data_gate_and_state_object(self):
        # Existing load_checkpoint restoration is already verified. Exercise only
        # the new routing/hash gate without saving or loading production weights.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'policy.json'; path.write_text(json.dumps(self.policy))
            data = {'annotation_sha256': 'pinned'}
            module = 'fashion_grounding.segmentation.resume_recipe.'
            with patch(module+'sha256', return_value=self.policy['parent_checkpoint_sha256']), \
                 patch(module+'load_checkpoint', return_value=self.state) as loader, \
                 patch(module+'audit_recipe', return_value={'audit': True}):
                state, audit = load_recipe_checkpoint('fake', self.b, data, path, 'head', ROOT)
                self.assertIs(state, self.state)
                loader.assert_called_once_with('fake', self.policy['parent_config_sha256'], data)
                self.assertTrue(audit['audit'])
            with patch(module+'sha256', return_value='wrong'), patch(module+'load_checkpoint') as loader:
                with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                    load_recipe_checkpoint('fake', self.b, data, path, 'head', ROOT)
                loader.assert_not_called()


class StageTests(unittest.TestCase):
    def setUp(self):
        self.plan = read('configs/runs/segmentation/min_scale_05_stage_plan.json')
        self.a = read(self.plan['control_report'])
        self.best = read(self.plan['prior_best_report'])
        self.b = copy.deepcopy(self.a)
        self.b.update(checkpoint_sha256='b'*64, training_git_commit='B', evaluation_git_commit='B')

    def test_comparison_keeps_old_best_or_promotes_b_by_overall_ap(self):
        self.assertEqual(compare_reports(self.a, self.b, self.best)['selected_group'], 'prior_best_18000')
        self.b['metrics']['AP'] = self.best['metrics']['AP']
        self.assertEqual(compare_reports(self.a, self.b, self.best)['selected_group'], 'prior_best_18000')
        self.b['metrics']['AP'] += 1
        result = compare_reports(self.a, self.b, self.best)
        self.assertEqual(result['selected_group'], 'trial_B_20000')
        self.assertAlmostEqual(result['B_minus_A_AP_points']['AP'], self.b['metrics']['AP']-self.a['metrics']['AP'])
        self.assertFalse(result['control_inference_repeated'])

    def test_partial_or_different_dev_is_ineligible(self):
        for key, value in [('complete', False), ('inference_errors', 1), ('evaluated_images', 249),
                           ('queries_read', True), ('evaluation_data', {}), ('completed_updates', 19000),
                           ('evaluation_config_sha256', 'wrong')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_reports(self.a, {**self.b, key: value}, self.best)

    def test_only_b_training_and_single_b_dev_command(self):
        commands = build_commands(self.plan, '/tmp/not-executed')
        self.assertEqual([p for p, _ in commands], ['train_B', 'dev250_B', 'compare_cached_A'])
        dev = commands[1][1]
        self.assertEqual(dev[dev.index('--checkpoints')+1:dev.index('--output')],
                         [str(Path('/tmp/not-executed/train/checkpoint_020000.pth').resolve())])
        self.assertNotIn('smoke', str(commands))
        self.assertEqual(self.plan['total_hard_limit_seconds'], 2700)

    def test_failed_training_stops_before_dev_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ, SEG_MIN_SCALE_WORKER=self.plan['run_id'], SEG_MIN_SCALE_DEADLINE='9999999'), \
             patch('run_min_scale_trial.arm_worker_deadline'), \
             patch('run_min_scale_trial.subprocess.call', return_value=7) as call:
            self.assertEqual(worker(self.plan, Path(temp)), 7)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(json.loads((Path(temp)/'phases.json').read_text())[0]['status'], 'failed')

    def test_expired_budget_never_starts_a_process(self):
        with patch('run_min_scale_trial.subprocess.Popen') as spawn:
            with self.assertRaises(TimeoutError):
                run_bounded(['not-started'], time.monotonic()-1)
            spawn.assert_not_called()

    def test_outer_deadline_stops_sleeping_process(self):
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'], started+0.2)
        self.assertLess(time.monotonic()-started, 3)
        self.assertEqual(run_bounded([sys.executable, '-c', 'raise SystemExit(7)'], time.monotonic()+3), 7)

    def test_worker_timer_requires_own_group_and_uses_absolute_deadline(self):
        with patch('run_min_scale_trial.os.getpid', return_value=1), \
             patch('run_min_scale_trial.os.getpgrp', return_value=2):
            with self.assertRaises(RuntimeError): arm_worker_deadline(time.monotonic()+10)
        with patch('run_min_scale_trial.os.getpid', return_value=1), \
             patch('run_min_scale_trial.os.getpgrp', return_value=1), \
             patch('run_min_scale_trial.time.monotonic', return_value=20), \
             patch('run_min_scale_trial.signal.signal'), \
             patch('run_min_scale_trial.signal.setitimer') as timer:
            arm_worker_deadline(30)
            self.assertEqual(timer.call_args.args[1], 10)
            with self.assertRaises(TimeoutError): arm_worker_deadline(19)


if __name__ == '__main__':
    unittest.main()
