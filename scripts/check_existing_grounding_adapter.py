"""CPU adapter checks against wyz's existing inference mathematics (synthetic features)."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import fashion_grounding
fashion_grounding.__path__.append("/root/autodl-tmp/fashion-grounding/workspaces/grounding/src/fashion_grounding")
from evaluate_existing_grounding import infer_scores
from fashion_grounding.grounding.heads import AppearanceResidual,padded_batch,batch_scores
torch.set_num_threads(1)
rng=np.random.default_rng(42)
v=rng.normal(size=(3,768)).astype("float32");v/=np.linalg.norm(v,axis=1,keepdims=True)
t=rng.normal(size=(1,768)).astype("float32");t/=np.linalg.norm(t,axis=1,keepdims=True)
base=np.array([2.,3.,1.],dtype="float32")
head=AppearanceResidual(768).eval()
assert np.array_equal(infer_scores(head,v,t,base),base)
path="/root/autodl-tmp/fashion-grounding/runs/grounding/M1-weak-train-20261112-r01/epoch_05.pt"
head.load_state_dict(torch.load(path,map_location="cpu")["model"])
head.eval()
sample={"visual":v,"text":t,"base":base,"target":0}
with torch.inference_mode():reference=batch_scores(head,padded_batch([sample],[0])).numpy()[0]
got=infer_scores(head,v,t,base)
assert np.array_equal(got,reference)
assert np.array_equal(got,infer_scores(head,v,t,base))
sample["target"]=2
with torch.inference_mode():other=batch_scores(head,padded_batch([sample],[0])).numpy()[0]
assert np.array_equal(got,other)
print("4 CPU checks passed: zero residual, saved-head formula parity, deterministic eval, target-independent scores.")
