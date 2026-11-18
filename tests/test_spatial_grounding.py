"""Hand-checkable geometry and text-conditioned M2 boundaries."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.geometry import geometry_features

HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
 import torch
 from fashion_grounding.grounding.spatial import SpatialResidual,make_head,residual,score_features

class GeometryTests(unittest.TestCase):
 def test_geometry_values_and_category_local_ranks(self):
  pack=CandidatePack(Path(__file__).resolve().parents[1]/"examples/candidates_gt.json",True)
  im,cs=pack.load(next(iter(pack.rows)));g=geometry_features(cs,im.size)
  x1,y1,x2,y2=cs[0].bbox_xyxy;w,h=im.size
  expected=[x1/w,y1/h,x2/w,y2/h,(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h,float(cs[0].mask.sum())/(w*h),np.log((x2-x1+1e-6)/(y2-y1+1e-6)),0,.5]
  np.testing.assert_allclose(g[0],expected,rtol=1e-6)
  np.testing.assert_array_equal(g[:,10],[0,1])
  np.testing.assert_array_equal(geometry_features(cs[:1],im.size)[:,10:],[ [.5,.5] ])
 def test_empty_geometry_and_bad_dimensions(self):
  self.assertEqual(geometry_features([],(100,100)).shape,(0,12))
  with self.assertRaisesRegex(ValueError,"dimensions"):geometry_features([],(0,100))

@unittest.skipUnless(HAS_TORCH,"requires grounding torch")
class SpatialTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)
 def test_same_appearance_initialization_and_zero_residual(self):
  torch.manual_seed(42);a=make_head("M1",4)
  torch.manual_seed(42);b=make_head("M2",4)
  for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.appearance.state_dict()[key]))
  v=torch.randn(2,3,4);t=torch.randn(2,4);g=torch.randn(2,3,12)
  self.assertTrue(torch.equal(b(v,t,g),torch.zeros(2,3)))
 def test_padding_does_not_change_valid_outputs(self):
  m=make_head("M2",4).eval()
  with torch.no_grad():m.spatial[-1].weight.normal_()
  v=torch.randn(1,1,4);t=torch.randn(1,4);g=torch.randn(1,1,12)
  expected=m(v,t,g)[0,0]
  padded=m(torch.cat([v,torch.zeros_like(v)],1),t,torch.cat([g,torch.zeros_like(g)],1))[0,0]
  torch.testing.assert_close(expected,padded)
 def test_geometry_text_interaction_learns_counterfactual(self):
  # Identical appearance: only query-conditioned geometry can solve both targets.
  m=make_head("M2",2)
  m.appearance.requires_grad_(False)
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
  v=torch.zeros(2,2,2);t=torch.eye(2);g=torch.zeros(2,2,12)
  g[:,0,10]=0;g[:,1,10]=1
  target=torch.tensor([0,1])
  for _ in range(80):
   opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(v,t,g),target);loss.backward();opt.step()
  m.eval()
  self.assertEqual(m(v,t,g).argmax(1).tolist(),[0,1])
  self.assertEqual(m(v,t.flip(0),g).argmax(1).tolist(),[1,0])
 def test_inference_has_no_target_argument_and_roundtrip(self):
  m=make_head("M2",2).eval();n=make_head("M2",2).eval();n.load_state_dict(m.state_dict())
  v=np.eye(2,dtype=np.float32);t=v[:1];g=np.zeros((2,12),np.float32);base=np.array([2.,1.],np.float32)
  np.testing.assert_array_equal(score_features(m,v,t,g,base),score_features(n,v,t,g,base))
if __name__=="__main__":unittest.main()
