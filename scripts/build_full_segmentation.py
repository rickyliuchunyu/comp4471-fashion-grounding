"""Build schema-3 full and delta views from the complete source disposition ledger."""
import ctypes
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from full_train_inventory import RUN, STAGED, PARENT, SHARED, SOURCE, NAME, REPO, PINS, sources, read_json, write_json, require, sha256
from fashion_grounding.data.schema import decode_mask, encode_mask
from build_staged_segmentation import verify_membership

DELTA = RUN/"fp_full_added_v1"
FULL = RUN/NAME
COMMON = {"schema_version":3, "illustrative_only":False}

def write_delta():
    DELTA.mkdir(exist_ok=True)
    for sub in ("images","masks"):
        (DELTA/sub).mkdir(exist_ok=True)
    data, by = sources()
    rules = {r["source_id"]:r for r in read_json(PARENT/"mapping_config.json")["fashionpedia"]}
    topology = ctypes.CDLL(str(RUN/"topology.so")).topology
    topology.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    def convert(inv):
        im = inv["source_image"]
        sid, w, h = im["id"], im["width"], im["height"]
        rel = "images/"+Path(inv["path"]).name
        shutil.copyfile(inv["path"],DELTA/rel)
        require(sha256(DELTA/rel)==inv["sha256"],"new image copy")
        manifest = dict(image_id=sid+1,file_name=rel,sha256=inv["sha256"],width=w,height=h,source_dataset="fashionpedia",source_split="official_train2020",source_image_id=str(sid),group_id=inv["group_id"],split="train")
        coco_im = {k:v for k,v in manifest.items() if k!="image_id"}
        coco_im.update(id=sid+1,license=im["license"],supervised_category_ids=list(range(1,9)))
        annotations, candidates, geometry = [], [], []
        for src in sorted(by[sid],key=lambda a:a["id"]):
            rule = rules[src["category_id"]]
            if rule["action"]=="exclude_annotation": continue
            require(rule["action"]=="map","unmapped or jumpsuit image")
            mask = decode_mask(src["segmentation"],h,w)
            seg = encode_mask(mask)
            require(np.array_equal(mu.decode(seg),mask),"source fidelity")
            x,y,bw,bh = [int(v) for v in mu.toBbox(seg)]
            area = int(mu.area(seg))
            require(area==int(mask.sum()) and bw>0 and bh>0,"area/box")
            aid=src["id"]+1
            annotations.append(dict(id=aid,image_id=sid+1,category_id=rule["project_category_id"],segmentation=seg,bbox=[x,y,bw,bh],area=area,iscrowd=src["iscrowd"],source_annotation_id=src["id"],source_category_id=src["category_id"],source_category_name=rule["source_name"],source_dataset="fashionpedia",attribute_ids=src.get("attribute_ids",[])))
            if not src["iscrowd"]:
                ref=f"masks/{aid}.json"
                write_json(DELTA/ref,seg)
                candidates.append(dict(candidate_id=f"fp-train-gt-{aid}",category_id=rule["project_category_id"],score=None,bbox_xyxy=[x,y,x+bw,y+bh],mask_reference=ref,mask_sha256=sha256(DELTA/ref),decoder_query_id=None,source_annotation_id=aid))
            crop = np.ascontiguousarray(mask[y:y+bh,x:x+bw],dtype=np.uint8)
            holes=ctypes.c_int()
            components=topology(crop.ctypes.data,bh,bw,ctypes.byref(holes))
            geometry.append([sid+1,aid,src["category_id"],components,holes.value])
        require(annotations,"empty training image")
        row = dict(image_id=sid+1,image_size_wh=[w,h],status="ok" if candidates else "no_candidates",candidates=candidates)
        return manifest,coco_im,annotations,row,geometry
    inventory=read_json(RUN/"eligible.json")
    ims=[]; ci=[]; anns=[]; candidates=[]; geom=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n,(im,c,aa,cc,gg) in enumerate(pool.map(convert,inventory),1):
            ims.append(im);ci.append(c);anns.extend(aa);candidates.append(cc);geom.extend(gg)
            if n%2000==0: print("verified new images",n,flush=True)
    shutil.copyfile(PARENT/"mapping_config.json",DELTA/"mapping_config.json")
    write_json(DELTA/"image_manifest.json",{**COMMON,"manifest_id":DELTA.name,"mapping_version":read_json(DELTA/"mapping_config.json")["mapping_version"],"images":ims})
    coco=read_json(PARENT/"annotations_coco.json")
    coco.update(images=ci,annotations=anns)
    write_json(DELTA/"annotations_coco.json",coco)
    write_candidates(DELTA,candidates)
    raw={k:data[k] for k in ("info","licenses","categories","attributes")}
    raw["images"]=[x["source_image"] for x in inventory]
    raw["annotations"]=[a for im in raw["images"] for a in by[im["id"]]]
    write_json(DELTA/"source_subset.json",raw)
    for f in ("queries_draft.jsonl","queries_accepted.jsonl"): (DELTA/f).write_text("")
    write_json(RUN/"topology_rows.json",geom)
    write_json(RUN/"source_fidelity.json",dict(verified_new_images=len(ims),verified_new_masks=len(anns),method="schema decode of official mask; exact equality with exported COCO RLE decode; area, bbox and candidate mask bytes checked",source_sha256=sha256(SOURCE),delta_annotation_sha256=sha256(DELTA/"annotations_coco.json"),delta_manifest_sha256=sha256(DELTA/"image_manifest.json")))

def write_candidates(folder,rows):
    obj={**COMMON,"candidate_set_id":folder.name+"-gt","candidate_source":"gt","image_manifest":{"path":"image_manifest.json","sha256":sha256(folder/"image_manifest.json")},"provenance":{"producer_run_id":RUN.name,"checkpoint_sha256":None,"decoder_config_sha256":None,"annotation_manifest_sha256":sha256(folder/"annotations_coco.json")},"images":rows}
    write_json(folder/"candidates_gt.json",obj)

def merge():
    FULL.mkdir(exist_ok=True)
    old_hashes=read_json(PARENT/"SHA256SUMS.json")
    asset_hashes={}
    for source in (PARENT,DELTA):
        for sub in ("images","masks"):
            (FULL/sub).mkdir(exist_ok=True)
            for p in (source/sub).iterdir():
                rel=sub+"/"+p.name
                require(rel not in asset_hashes,"overlapping asset")
                shutil.copyfile(p,FULL/rel)
                value=sha256(FULL/rel)
                expected=old_hashes[rel] if source==PARENT else sha256(p)
                require(value==expected,"asset preservation")
                asset_hashes[rel]=value
    for name,fields in [("image_manifest",["images"]),("annotations_coco",["images","annotations"]),("source_subset",["images","annotations"]),("candidates_gt",["images"])]:
        file=name+".json";p=read_json(PARENT/file);d=read_json(DELTA/file)
        for key in fields:p[key].extend(d[key])
        if name=="image_manifest":p["manifest_id"]=NAME
        if name=="candidates_gt":write_candidates(FULL,p["images"])
        else:write_json(FULL/file,p)
    verify_membership(PARENT,DELTA,FULL)
    for src,label in [(PARENT,"old"),(DELTA,"added")]:
        c=read_json(src/"annotations_coco.json")
        write_json(FULL/(label+"_subset_membership.json"),{"image_ids":[i["id"] for i in c["images"]],"annotation_ids":[a["id"] for a in c["annotations"]]})
    for f in ("mapping_config.json","queries_draft.jsonl","queries_accepted.jsonl"):shutil.copyfile(PARENT/f,FULL/f)
    write_json(RUN/"asset_hashes.json",asset_hashes)
    print("parent records and assets preserved",flush=True)

if __name__=="__main__":
    write_delta()
    merge()
