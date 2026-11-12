"""Planned M1 appearance residual; frozen encoder features are its only input."""
import torch
from torch import nn

class AppearanceResidual(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension=dimension
        self.layers=nn.Sequential(nn.Linear(3*dimension,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,visual,text):
        # visual B,N,D; text B,D. No category, query type or target input.
        text=text[:,None,:].expand_as(visual)
        return self.layers(torch.cat((visual,text,visual*text),dim=-1)).squeeze(-1)

def padded_batch(samples,indices):
    chosen=[samples[i] for i in indices]
    n=max(len(s["visual"]) for s in chosen);d=chosen[0]["visual"].shape[1]
    v=torch.zeros(len(chosen),n,d);t=torch.zeros(len(chosen),d)
    base=torch.zeros(len(chosen),n);valid=torch.zeros(len(chosen),n,dtype=torch.bool)
    labels=torch.empty(len(chosen),dtype=torch.long)
    for b,s in enumerate(chosen):
        count=len(s["visual"])
        if count<1:raise ValueError("empty candidates cannot enter padded softmax")
        v[b,:count]=torch.from_numpy(s["visual"]);t[b]=torch.from_numpy(s["text"][0])
        base[b,:count]=torch.from_numpy(s["base"]);valid[b,:count]=True
        labels[b]=s["target"]
        if not 0<=labels[b]<count:raise ValueError("target outside candidates")
    return v,t,base,valid,labels

def batch_scores(model,batch):
    v,t,base,valid,_=batch
    return (base+model(v,t)).masked_fill(~valid,float("-inf"))

def ranking_loss(scores,batch):
    valid=batch[3];targets=batch[4]
    rows=valid.sum(1)>1
    if not rows.any():raise ValueError("no multi-candidate ranking examples")
    return nn.functional.cross_entropy(scores[rows],targets[rows])
