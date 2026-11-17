"""M2 adds only a text-conditioned geometry branch to the identical M1 head."""
import torch
from torch import nn
from .heads import AppearanceResidual

class SpatialResidual(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.dimension=dimension
        self.appearance=AppearanceResidual(dimension)
        self.geometry=nn.Sequential(nn.Linear(12,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.spatial=nn.Sequential(nn.Linear(dimension+64,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.spatial[-1].weight);nn.init.zeros_(self.spatial[-1].bias)
    def forward(self,visual,text,geometry):
        embedded=self.geometry(geometry)
        t=text[:,None,:].expand(*embedded.shape[:-1],self.dimension)
        spatial=self.spatial(torch.cat((t,embedded),dim=-1)).squeeze(-1)
        return self.appearance(visual,text)+spatial

def make_head(method,dimension):
    if method=="M1":return AppearanceResidual(dimension)
    if method=="M2":return SpatialResidual(dimension)
    raise ValueError("unknown learned ranker")

def residual(model,visual,text,geometry):
    return model(visual,text,geometry) if isinstance(model,SpatialResidual) else model(visual,text)

def score_features(model,visual,text,geometry,base):
    """Inference boundary: only frozen features and candidate geometry, never labels."""
    with torch.inference_mode():
        scores=base+residual(model,torch.from_numpy(visual)[None],
            torch.from_numpy(text),torch.from_numpy(geometry)[None]).numpy()[0]
    if not torch.isfinite(torch.from_numpy(scores)).all():raise ValueError("nonfinite ranker scores")
    return scores
