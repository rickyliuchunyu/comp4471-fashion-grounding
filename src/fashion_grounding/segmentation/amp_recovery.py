"""Bounded AMP overflow replay; failed attempts never count as optimizer updates."""
import math

import torch

from .training_state import capture_rng, restore_rng


def validate_policy(policy):
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {
            'max_retries_per_update', 'max_overflow_events', 'min_loss_scale'}:
        raise ValueError('Invalid AMP recovery policy')
    for key in ('max_retries_per_update', 'max_overflow_events'):
        if type(policy[key]) is not int or not 0 < policy[key] <= 32:
            raise ValueError('AMP recovery must have bounded positive retry limits')
    if (type(policy['min_loss_scale']) not in (float, int)
            or not math.isfinite(policy['min_loss_scale']) or policy['min_loss_scale'] <= 0):
        raise ValueError('Invalid minimum AMP loss scale')


class AmpRecovery:
    def __init__(self, policy, emit, state=None):
        validate_policy(policy)
        self.policy, self.emit = policy, emit
        self.overflow_events = (state or {}).get('overflow_events', 0)
        if type(self.overflow_events) is not int or self.overflow_events < 0:
            raise ValueError('Invalid saved AMP recovery count')
        if policy and self.overflow_events > policy['max_overflow_events']:
            raise ValueError('Saved AMP overflow count exceeds policy')

    def state_dict(self):
        return dict(overflow_events=self.overflow_events)

    def backward(self, model, optimizer, scaler, sampler, forward_backward, update, check_budget):
        """Return finite, unscaled gradients; caller alone performs the successful step.

        forward_backward consumes data and accumulates one effective batch, rejecting
        non-finite forward losses. Replay restores sampler, all RNGs and model buffers,
        so augmentation, point sampling, dropout and batch statistics are repeated.
        The scaler's backoff is deliberately retained. No CUDA/model smoke is involved.
        """
        snapshot = None
        if self.policy:
            snapshot = (sampler.state_dict(), capture_rng(),
                        [(buffer, buffer.detach().clone()) for buffer in model.buffers()])
        retries = 0
        while True:
            check_budget()
            optimizer.zero_grad(set_to_none=True)
            details = forward_backward()
            scaler.unscale_(optimizer)
            bad = [name for name, param in model.named_parameters()
                   if param.grad is not None and not torch.isfinite(param.grad).all()]
            if not bad:
                return details, retries
            self.overflow_events += 1
            before = scaler.get_scale()
            event = dict(attempted_update=update, completed_updates=update-1,
                         attempt=retries+1, loss_scale=before,
                         nonfinite_gradient_parameters=bad, **details,
                         overflow_events=self.overflow_events)
            reason = None
            if not self.policy or not scaler.is_enabled():
                reason = 'AMP recovery not enabled'
            elif retries >= self.policy['max_retries_per_update']:
                reason = 'Per-update AMP retry limit reached'
            elif self.overflow_events > self.policy['max_overflow_events']:
                reason = 'Total AMP overflow limit reached'
            elif before * scaler.get_backoff_factor() < self.policy['min_loss_scale']:
                reason = 'Minimum AMP loss scale reached'
            if reason:
                self.emit(dict(event, action='stop', reason=reason))
                raise ValueError(reason+'; no optimizer/scheduler step performed')
            # unscale_ recorded the inf/NaN. Public GradScaler.step skips optimizer.step;
            # update lowers the scale and resets per-optimizer state for the next attempt.
            scaler.step(optimizer)
            scaler.update()
            after = scaler.get_scale()
            if not math.isfinite(after) or not 0 < after < before:
                self.emit(dict(event, action='stop', loss_scale_after=after,
                               reason='AMP did not reduce loss scale'))
                raise ValueError('AMP did not reduce loss scale')
            sampler.load_state_dict(snapshot[0])
            restore_rng(snapshot[1])
            with torch.no_grad():
                for buffer, saved in snapshot[2]:
                    buffer.copy_(saved)
            retries += 1
            self.emit(dict(event, action='replay_same_effective_batch', loss_scale_after=after,
                           optimizer_step_skipped=True, scheduler_advanced=False,
                           sampler_rng_buffers_restored=True))
