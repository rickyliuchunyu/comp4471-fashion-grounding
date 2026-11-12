"""One evaluator for GT, predicted, and external candidates; failures keep denominator."""
from collections import Counter,defaultdict
from fashion_grounding.data.schema import require,unique_index
from fashion_grounding.data.candidates import finite
from .matching import associate,mask_iou

def evaluate(queries, predictions, candidate_package, candidate_masks, annotations, gt_masks):
    require(bool(queries),"empty evaluation query set")
    qs=unique_index(queries,"query_id");ps=unique_index(predictions,"query_id")
    require(set(qs)==set(ps),"missing or unknown predictions: incomplete run")
    require(len({p["run_id"] for p in predictions})==1 and len({p["method_id"] for p in predictions})==1,"mixed run/method")
    require(all(p["run_id"] and p["method_id"] for p in predictions),"run/method required")
    anns=unique_index(annotations,"id");images=unique_index(candidate_package["images"],"image_id")
    source=candidate_package["candidate_source"];matches={}
    for iid,im in images.items():
        aa=[a for a in annotations if a["image_id"]==iid and not a["iscrowd"]]
        if source=="gt":
            matches[iid]={}
            for c in im["candidates"]:
                aid=c["source_annotation_id"]
                require(aid in anns and anns[aid]["image_id"]==iid and not anns[aid]["iscrowd"] and anns[aid]["category_id"]==c["category_id"],"GT reference mismatch")
                require(mask_iou(candidate_masks[c["candidate_id"]],gt_masks[aid])==1,"GT mask mismatch")
                matches[iid][c["candidate_id"]]=aid
        elif source=="predicted":matches[iid]=associate(im["candidates"],aa,candidate_masks,gt_masks)
        else:matches[iid]={}
    rows=[]
    for qid,q in qs.items():
        require(q["review_status"]=="accepted","unreviewed query")
        require(q["target_instance_id"] in anns,"missing target")
        target=anns[q["target_instance_id"]]
        require(target["image_id"]==q["image_id"] and not target["iscrowd"],"target image/crowd")
        p=ps[qid];im=images[q["image_id"]];cs={c["candidate_id"]:c for c in im["candidates"]}
        require(p["schema_version"]==3 and p["illustrative_only"] is False,"prediction schema")
        for key in ("image_id","query","language","split"):require(p[key]==q[key],"prediction query metadata mismatch")
        require(p["candidate_source"]==source and p["candidate_set_id"]==candidate_package["candidate_set_id"],"candidate provenance mismatch")
        status=p["status"];selected=p["selected_candidate"]
        require(status in ("success","no_candidates","no_output","rule_unsupported","rejected","inference_error"),"prediction status")
        if im["status"]=="inference_error":require(status=="inference_error","upstream error not propagated")
        if im["status"]=="no_candidates":require(status=="no_candidates","empty candidates not propagated")
        if status=="no_candidates":require(im["status"]=="no_candidates","false no_candidates")
        if status=="inference_error":require(isinstance(p["error"],dict) and p["error"].get("code") and p["error"].get("message"),"error details required")
        else:require(p["error"] is None,"unexpected error")
        cid=None
        if status=="success":
            require(isinstance(selected,dict) and selected["candidate_id"] in cs,"unknown selected candidate")
            require(selected["grounding_score"] is None or finite(selected["grounding_score"]),"non-finite grounding score")
            cid=selected["candidate_id"]
        else:require(selected is None,"failure with selected output")
        target_mask=gt_masks[target["id"]]
        overlaps={k:mask_iou(candidate_masks[k],target_mask) for k in cs}
        covered=target["id"] in matches[q["image_id"]].values()
        correct=bool(cid is not None and matches[q["image_id"]].get(cid)==target["id"]) if source!="external" else None
        rows.append({"query_id":qid,"semantic_query_id":q["semantic_query_id"],"image_id":q["image_id"],
            "language":q["language"],"category_id":target["category_id"],"source_category":target.get("source_category_name"),
            "query_types":q["query_types"],"same_category_instance_count":q.get("same_category_instance_count"),
            "same_source_category_instance_count":q.get("same_source_category_instance_count"),
            "candidate_count":len(cs),"target_area":target["area"],
            "contrast_set_id":q.get("contrast_set_id"),"paraphrase_group_id":q.get("paraphrase_group_id"),
            "status":status,"selected_candidate_id":cid,"target_instance_id":target["id"],
            "correct":correct,"covered":covered,"geometric_recalled":max(overlaps.values(),default=0)>=.5,
            "target_mask_iou":overlaps.get(cid,0),"success_at_0_5":overlaps.get(cid,0)>=.5})
    def summary(rr):
        n=len(rr);coverage=sum(r["covered"] for r in rr)
        correct=sum(r["correct"] is True for r in rr)
        return {"queries":n,"images":len({r["image_id"] for r in rr}),
            "oracle_selection_accuracy":correct/n if n and source=="gt" else None,
            "predicted_selection_accuracy":correct/n if n and source=="predicted" else None,
            "conditional_selection_accuracy":sum(r["correct"] is True and r["covered"] for r in rr)/coverage if coverage and source=="predicted" else None,
            "matched_candidate_coverage":coverage/n if n and source=="predicted" else None,
            "geometric_recall_at_0_5":sum(r["geometric_recalled"] for r in rr)/n if n else None,
            "mean_target_mask_iou":sum(r["target_mask_iou"] for r in rr)/n if n else None,
            "success_at_0_5":sum(r["success_at_0_5"] for r in rr)/n if n else None,
            "status_counts":dict(Counter(r["status"] for r in rr))}
    groups={}
    for field in ("language","category_id","source_category","query_types","candidate_count","same_category_instance_count","same_source_category_instance_count","area_bin"):
        buckets=defaultdict(list)
        for row in rows:
            value=("small" if row["target_area"]<32**2 else "medium" if row["target_area"]<96**2 else "large") if field=="area_bin" else row[field]
            for key in value if isinstance(value,list) else [value]:buckets[str(key)].append(row)
        groups[field]={k:summary(v) for k,v in sorted(buckets.items())}
    metric="oracle_selection_accuracy" if source=="gt" else "predicted_selection_accuracy"
    categories=[v[metric] for v in groups["category_id"].values() if v[metric] is not None]
    result=summary(rows);result["macro_category_accuracy"]=sum(categories)/len(categories) if categories else None
    pairs={}
    for field in ("semantic_query_id","contrast_set_id","paraphrase_group_id"):
        buckets=defaultdict(list)
        for r in rows:
            if r[field]:
                # D1/D2 are evaluated within each language, avoiding translation inflation.
                key=(r[field],None if field=="semantic_query_id" else r["language"])
                buckets[key].append(r)
        valid=[rr for rr in buckets.values() if len(rr)>=2]
        pairs[field]={"groups":len(valid),"all_correct_rate":sum(all(r["correct"] is True for r in rr) for rr in valid)/len(valid) if valid and source!="external" else None}
        if field=="semantic_query_id":
            pairs[field]["selection_consistency"]=sum(len({r["selected_candidate_id"] for r in rr})==1 and rr[0]["selected_candidate_id"] is not None for rr in valid)/len(valid) if valid else None
    return {"candidate_source":source,"summary":result,"groups":groups,"paired":pairs,"rows":rows,
            "association":{str(iid):m for iid,m in matches.items()}}
