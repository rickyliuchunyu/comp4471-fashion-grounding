"""Evaluate a complete frozen query list; missing outputs fail closed."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,safe_path,unique_index,validate_queries,decode_mask
from fashion_grounding.data.candidates import load_candidates
from fashion_grounding.evaluation.metrics import evaluate

def evaluate_files(queries_path,predictions_path,candidates_path,annotations_path,output,allow_train=False,*,allow_draft=False):
    require(not output.exists(),"evaluation output exists")
    package,manifest,masks=load_candidates(candidates_path)
    coco=read_json(annotations_path)
    require(coco["schema_version"]==3 and coco["illustrative_only"] is False,"real GT annotations required")
    images=unique_index(manifest["images"],"image_id");gt_images=unique_index(coco["images"],"id")
    require(set(images)==set(gt_images),"GT/candidate image coverage mismatch")
    for iid,im in images.items():
        gt_image = gt_images[iid]
        keys = ("width", "height", "source_image_id", "source_dataset")
        if allow_draft and im.get("source_dataset") == "mhp_v2" and not any(k in gt_image for k in keys[2:]):
            # The official MHP COCO image rows carry filename and dimensions but omit source fields.
            keys = ("width", "height", "file_name")
        require(all(k in im and k in gt_image and im[k] == gt_image[k] for k in keys), "GT/candidate image mismatch")
    anns=unique_index(coco["annotations"],"id")
    qs=[json.loads(s) for s in queries_path.read_text().splitlines() if s.strip()]
    validate_queries(qs,images,anns,accepted_only=not allow_draft)
    require(not allow_draft or all(q["split"] == "dev" and q["review_status"] == "needs_revision" for q in qs),"draft evaluation accepts only unreviewed development queries")
    require(len({q["split"] for q in qs})==1,"mixed splits")
    require(all(q["split"] in ("dev","test") or (allow_train and q["split"]=="train") for q in qs),"train diagnostic requires explicit flag")
    predictions=[json.loads(s) for s in predictions_path.read_text().splitlines() if s.strip()]
    for p in predictions:
        ref=p["candidate_manifest"]
        require(ref["sha256"]==sha256(candidates_path),"prediction candidate hash mismatch")
        require(sha256(safe_path(predictions_path.parent,ref["path"]))==ref["sha256"],"prediction reference differs")
    if package["candidate_source"]=="gt":
        require(package["provenance"]["annotation_manifest_sha256"]==sha256(annotations_path),"GT annotation hash mismatch")
    gt_masks={aid:decode_mask(a["segmentation"],images[a["image_id"]]["height"],images[a["image_id"]]["width"]) for aid,a in anns.items()}
    result=evaluate(qs,predictions,package,masks,list(anns.values()),gt_masks,allow_draft=allow_draft)
    result["provenance"]={"queries_sha256":sha256(queries_path),"predictions_sha256":sha256(predictions_path),"candidates_sha256":sha256(candidates_path),"annotations_sha256":sha256(annotations_path),"split":qs[0]["split"],"scope":"unreviewed_dev_diagnostic" if allow_draft else "training_pilot_diagnostic_not_generalization" if qs[0]["split"]=="train" else "evaluation"}
    write_json(output,result)
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("queries","predictions","candidates","annotations","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--allow-train-diagnostic",action="store_true");a=p.parse_args()
    print(json.dumps(evaluate_files(a.queries,a.predictions,a.candidates,a.annotations,a.output,a.allow_train_diagnostic)["summary"],indent=2))
