"""Meaningful M1 optimization boundaries; synthetic features are not results."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from fashion_grounding.grounding.heads import AppearanceResidual,padded_batch,batch_scores,ranking_loss
    from fashion_grounding.grounding.weak_trial import fit_epoch,checkpoint,restore,measure

@unittest.skipUnless(HAS_TORCH,"M1 tests require grounding's torch environment")
class RankingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.samples=[dict(visual=np.eye(2,dtype=np.float32),text=np.array([[1.,0.]],np.float32),
                           base=np.array([1.,0.],np.float32),target=1),
                      dict(visual=np.array([[1.,0.]],np.float32),text=np.array([[1.,0.]],np.float32),
                           base=np.array([1.],np.float32),target=0)]
        self.model=AppearanceResidual(2)
    def test_initial_scores_preserve_baseline_and_padding(self):
        b=padded_batch(self.samples,[0,1]);scores=batch_scores(self.model,b)
        np.testing.assert_array_equal(scores[0].detach(),[1.,0.])
        self.assertEqual(scores[1,0],1)
        self.assertTrue(torch.isneginf(scores[1,1]))
    def test_loss_excludes_singletons_and_matches_hand_value(self):
        b=padded_batch(self.samples,[0,1]);loss=ranking_loss(batch_scores(self.model,b),b)
        self.assertAlmostEqual(loss.item(),float(np.log1p(np.e)),places=6)
        b=padded_batch(self.samples,[1])
        with self.assertRaisesRegex(ValueError,"multi-candidate"):ranking_loss(batch_scores(self.model,b),b)
    def test_padding_does_not_change_valid_scores(self):
        self.model.eval()
        with torch.no_grad():self.model.layers[-1].weight.normal_()
        one=batch_scores(self.model,padded_batch(self.samples,[1]))[0,0]
        many=batch_scores(self.model,padded_batch(self.samples,[0,1]))[1,0]
        torch.testing.assert_close(one,many)
    def test_optimizer_only_updates_head_and_can_fit_counterexample(self):
        before=copy.deepcopy(self.samples)
        opt=torch.optim.AdamW(self.model.parameters(),lr=.01)
        initial=measure(self.model,self.samples,2)[0]["multi_candidate_cross_entropy"]
        for _ in range(20):fit_epoch(self.model,opt,self.samples,2)
        final=measure(self.model,self.samples,2)[0]
        self.assertLess(final["multi_candidate_cross_entropy"],initial*.2)
        self.assertEqual(final["correct_against_weak_labels"],2)
        for old,new in zip(before,self.samples):
            np.testing.assert_array_equal(old["visual"],new["visual"])
            np.testing.assert_array_equal(old["text"],new["text"])
    def test_resume_reproduces_next_optimizer_update(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=.001)
        fit_epoch(self.model,opt,self.samples,2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.pt"
            torch.save(checkpoint(self.model,opt,1,1,{},{}),path)
            expected_log=fit_epoch(self.model,opt,self.samples,2)
            expected=copy.deepcopy(self.model.state_dict())
            restored=AppearanceResidual(2);other_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
            restore(torch.load(path),restored,other_opt)
            actual_log=fit_epoch(restored,other_opt,self.samples,2)
        self.assertEqual(expected_log,actual_log)
        for k,value in expected.items():self.assertTrue(torch.equal(value,restored.state_dict()[k]))
    def test_out_of_range_target_and_empty_candidates_rejected(self):
        samples=copy.deepcopy(self.samples);samples[0]["target"]=2
        with self.assertRaisesRegex(ValueError,"target"):padded_batch(samples,[0])
        samples[0]["visual"]=np.empty((0,2),np.float32)
        with self.assertRaisesRegex(ValueError,"empty"):padded_batch(samples,[0])
if __name__=="__main__":unittest.main()
