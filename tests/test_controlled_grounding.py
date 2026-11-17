"""Controlled experiment ordering, split isolation, supervision and M2 resume."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
 import torch
 from fashion_grounding.grounding.controlled import training_order,make_batch,training_scores,manifest_separation,supervised_samples
 from fashion_grounding.grounding.heads import ranking_loss
 from fashion_grounding.grounding.spatial import make_head
@unittest.skipUnless(HAS_TORCH,"requires grounding torch")
class ControlledTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1)
  self.samples=[{"visual":np.eye(2,dtype=np.float32),"text":np.array([[1.,0.]],np.float32),
                 "base":np.array([1.,0.],np.float32),"geometry":np.zeros((2,12),np.float32),"target":1}
                for _ in range(5)]
 def test_full_batch_update_budget_order_independent_of_model_rng(self):
  first=training_order(self.samples,42,7,3)
  torch.manual_seed(789);torch.randn(100)
  self.assertEqual(first,training_order(self.samples,42,7,3))
  self.assertEqual(len(first),21);self.assertEqual(sorted(first[:5]),list(range(5)))
  self.samples.append({"visual":np.ones((1,2),np.float32)})
  self.assertNotIn(5,training_order(self.samples,42,7,3))
 def test_m2_resume_preserves_next_update(self):
  torch.manual_seed(42);model=make_head("M2",2);opt=torch.optim.AdamW(model.parameters(),lr=.001)
  batch=make_batch(self.samples,[0,1,2])
  def update(m,o):
   m.train();o.zero_grad();loss=ranking_loss(training_scores(m,batch),batch);loss.backward();o.step()
  update(model,opt);state=copy.deepcopy(model.state_dict());optim=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
  update(model,opt);expected=copy.deepcopy(model.state_dict())
  restored=make_head("M2",2);restored.load_state_dict(state)
  other=torch.optim.AdamW(restored.parameters(),lr=.001);other.load_state_dict(optim);torch.set_rng_state(rng)
  update(restored,other)
  for k,v in expected.items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))
 def test_cross_split_hash_overlap_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   paths={}
   for i,name in enumerate(("clean1000","expanded1149","dev","challenge","test")):
    p=Path(tmp)/name;p.mkdir();paths[name]=str(p)
    j=0 if name=="expanded1149" else i
    (p/"image_manifest.json").write_text(json.dumps({"images":[{"image_id":j,"sha256":str(j),"group_id":str(j)}]}))
   self.assertEqual(len(manifest_separation(paths)),18)
   path=Path(paths["dev"])/"image_manifest.json"
   data=json.loads(path.read_text());data["images"][0]["sha256"]="0";path.write_text(json.dumps(data))
   with self.assertRaisesRegex(ValueError,"overlap"):manifest_separation(paths)
if __name__=="__main__":unittest.main()
