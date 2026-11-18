"""Twelve original-image geometry features, using candidate information only."""
import numpy as np
from fashion_grounding.data.schema import require
from .rules import category_ranks

def geometry_features(candidates,image_size_wh):
    width,height=image_size_wh
    require(width>0 and height>0,"invalid image dimensions")
    rows=[]
    for c in candidates:
        x1,y1,x2,y2=c.bbox_xyxy
        require(0<=x1<x2<=width and 0<=y1<y2<=height,"invalid geometry box")
        require(c.mask.shape==(height,width),"geometry mask dimensions")
        w,h=x2-x1,y2-y1
        rows.append([x1/width,y1/height,x2/width,y2/height,
            (x1+x2)/(2*width),(y1+y2)/(2*height),w/width,h/height,
            float(c.mask.sum())/(width*height),np.log((w+1e-6)/(h+1e-6)),0.,0.])
    result=np.asarray(rows,dtype=np.float32).reshape(-1,12)
    result[:,10]=category_ranks(candidates,0);result[:,11]=category_ranks(candidates,1)
    require(np.isfinite(result).all(),"nonfinite geometry")
    return result
