"""Evaluation-only same-class greedy one-to-one association, fixed at IoU >= .5."""
import numpy as np

def mask_iou(a,b):
    if a.shape != b.shape: raise ValueError("mask shape mismatch")
    union=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/union) if union else 0.0

def associate(candidates, annotations, candidate_masks, gt_masks):
    available={a["id"]:a for a in annotations if not a["iscrowd"]}
    matches={}
    for c in sorted(candidates,key=lambda c:(-c["score"],c["candidate_id"])):
        options=[(mask_iou(candidate_masks[c["candidate_id"]],gt_masks[aid]),aid)
                 for aid,a in available.items() if a["category_id"]==c["category_id"]]
        if not options: continue
        overlap,aid=min(options,key=lambda x:(-x[0],x[1]))
        if overlap >= .5:
            matches[c["candidate_id"]]=aid
            del available[aid]
    return matches
