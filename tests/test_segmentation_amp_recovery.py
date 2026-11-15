"""New overflow/replay paths only; no dataset, Mask2Former or legacy smoke."""
import copy
import os
import random
import unittest

import numpy as np
import torch

from fashion_grounding.segmentation.amp_recovery import AmpRecovery, validate_policy
from fashion_grounding.segmentation.training_state import (
    StatefulSampler, UpdateSchedule, capture_rng, restore_rng, state_digest)


POLICY = dict(max_retries_per_update=4, max_overflow_events=16, min_loss_scale=0.0001)
DEVICE = os.environ.get('AMP_TEST_DEVICE', 'cpu')


def make_scaler(scale=8):
    if DEVICE == 'cuda':
        return torch.cuda.amp.GradScaler(init_scale=scale, growth_interval=2000)
    return torch.amp.GradScaler('cpu', init_scale=scale, growth_interval=2000)


class AmpRecoveryTests(unittest.TestCase):
    def setUp(self):
        random.seed(71); np.random.seed(71); torch.manual_seed(71)
        self.model = torch.nn.Linear(2, 1).to(DEVICE)
        self.model.register_buffer('forward_count', torch.tensor(0., device=DEVICE))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.schedule = UpdateSchedule(self.opt, dict(warmup_updates=0, warmup_start_factor=1.,
                                                     lr_milestones=[1], lr_decay=0.1))
        self.sampler = StatefulSampler(3, 71)
        self.scaler = make_scaler()
        self.events, self.attempts = [], []

    def forward(self, inject='first'):
        indices, values = [], []
        for _ in range(2):  # Exercise effective-batch accumulation and epoch crossing.
            index = self.sampler.next()
            x = torch.randn(2, 2, device=DEVICE) + random.random() + np.random.rand()
            self.model.forward_count.add_(1)
            loss = (self.model(torch.nn.functional.dropout(x, p=0.2))-index).square().mean()/2
            self.scaler.scale(loss).backward()
            indices.append(index); values.append(float(loss.detach()))
        if inject == 'always' or (inject == 'first' and not self.attempts):
            next(self.model.parameters()).grad.view(-1)[0] = float('inf')
        self.attempts.append((indices, values))
        return dict(image_ids=indices, instance_counts=[1, 1], loss_components={'loss': sum(values)})

    def run_backward(self, policy=POLICY, inject='first', state=None, check_budget=lambda: None):
        self.recovery = AmpRecovery(policy, self.events.append, state)
        return self.recovery.backward(self.model, self.opt, self.scaler, self.sampler,
                                     lambda: self.forward(inject), 1, check_budget)

    def test_actual_scaler_skips_optimizer_and_replays_same_rng_sampler_buffers(self):
        before = copy.deepcopy(dict(model=self.model.state_dict(), sampler=self.sampler.state_dict(), rng=capture_rng()))
        details, retries = self.run_backward()
        self.assertEqual(retries, 1)
        self.assertEqual(self.attempts[0], self.attempts[1])
        self.assertEqual(self.model.forward_count.item(), 2)
        self.assertEqual(self.schedule.completed_updates, 0)
        self.assertFalse(self.opt.state)  # AdamW step/moments were not created by overflow.
        for key in ('weight', 'bias'):
            self.assertTrue(torch.equal(self.model.state_dict()[key], before['model'][key]))
        recovered_grad = [p.grad.clone() for p in self.model.parameters()]
        after_rng, after_sampler = state_digest(capture_rng()), state_digest(self.sampler.state_dict())
        self.assertEqual(self.scaler.get_scale(), 4)
        self.assertEqual(self.events[0]['action'], 'replay_same_effective_batch')
        self.model.load_state_dict(before['model']); self.sampler.load_state_dict(before['sampler'])
        restore_rng(before['rng']); self.scaler = make_scaler(4); self.opt.zero_grad(set_to_none=True)
        expected = self.forward(inject='none'); self.scaler.unscale_(self.opt)
        self.assertEqual(details, expected)
        for a, p in zip(recovered_grad, self.model.parameters()):
            self.assertTrue(torch.equal(a, p.grad))
        self.assertEqual(after_rng, state_digest(capture_rng()))
        self.assertEqual(after_sampler, state_digest(self.sampler.state_dict()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
        self.scaler.step(self.opt); self.scaler.update(); self.schedule.step()
        self.assertEqual(self.schedule.completed_updates, 1)
        self.assertTrue(all(s['step'].item() == 1 for s in self.opt.state.values()))
        self.assertEqual(AmpRecovery(POLICY, lambda x: None, self.recovery.state_dict()).overflow_events, 1)

    def test_persistent_overflow_stops_at_per_update_limit(self):
        with self.assertRaisesRegex(ValueError, 'Per-update'):
            self.run_backward({**POLICY, 'max_retries_per_update': 1}, inject='always')
        self.assertEqual(len(self.attempts), 2)
        self.assertFalse(self.opt.state)
        self.assertEqual(self.schedule.completed_updates, 0)
        self.assertEqual(self.events[-1]['action'], 'stop')

    def test_total_limit_and_scale_floor_stop_without_update(self):
        for policy, state, reason in [
                (POLICY, {'overflow_events': 16}, 'Total'),
                ({**POLICY, 'min_loss_scale': 8}, None, 'Minimum'),
                (None, None, 'not enabled')]:
            self.setUp()
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.run_backward(policy, state=state)
            self.assertFalse(self.opt.state)
            self.assertEqual(self.schedule.completed_updates, 0)

    def test_deadline_is_rechecked_before_replay(self):
        calls = []
        def check():
            calls.append(1)
            if len(calls) > 1: raise TimeoutError('test deadline')
        with self.assertRaises(TimeoutError): self.run_backward(check_budget=check)
        self.assertEqual(len(self.attempts), 1)
        self.assertFalse(self.opt.state)

    def test_nonfinite_forward_failure_is_not_retried(self):
        attempts = []
        def forward():
            attempts.append(1)
            raise ValueError('Non-finite or missing loss')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            AmpRecovery(POLICY, self.events.append).backward(self.model, self.opt, self.scaler,
                self.sampler, forward, 1, lambda: None)
        self.assertEqual(len(attempts), 1); self.assertEqual(self.events, [])

    def test_policy_rejects_unbounded_or_invalid_values(self):
        for changes in [{'max_retries_per_update': 0}, {'max_retries_per_update': True},
                        {'max_overflow_events': 100000}, {'min_loss_scale': float('nan')},
                        {'min_loss_scale': -1}, {'extra': 1}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_policy({**POLICY, **changes})


if __name__ == '__main__': unittest.main()
