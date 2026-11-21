"""Summarize and publish the complete source inventory with hash-bound QA evidence."""
import csv
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from full_train_inventory import RUN, NAME, SHARED, STAGED, PARENT, SOURCE, REPO, PINS, read_json, write_json, require, sha256
from build_full_segmentation import DELTA, FULL

def counts(rows,groups):
    images={a["image_id"] for a in rows}
    return dict(images=len(images),groups=len({groups[i] for i in images}),instances=len(rows))

def report(root):
    coco=read_json(root/"annotations_coco.json")
    groups={i["id"]:i["group_id"] for i in coco["images"]}
    rows=coco["annotations"]
    tables={}
    def bins(label,annotations):
        tables[label]={name:counts([a for a in annotations if a["area"]<=limit],groups) for name,limit in [("all",float("inf")),("small",1024),("extreme",256)]}
    bins("overall",rows)
    for cid in range(1,9):bins("project_"+str(cid),[a for a in rows if a["category_id"]==cid])
    for cid in range(27):
        if cid!=11:bins("source_"+str(cid),[a for a in rows if a["source_category_id"]==cid])
    for cid,name in [(None,"all"),(23,"shoe"),(24,"bag")]:
        tables["source_cutout_"+name]=counts([a for a in rows if 308 in a["attribute_ids"] and (cid is None or a["source_category_id"]==cid)],groups)
    return tables

def record_bridges(ledger):
    audit=read_json(RUN/"duplicates.json")
    components={tuple(sorted(r["component"])) for r in audit["excluded"] if "component" in r}
    parent={"train-"+i for i,state in ledger.items() if state=="parent"}
    frozen={r["key"] for r in read_json(STAGED/"protected_fingerprints.json")}
    val=set()
    for name in ["fp_dev250_v1","fp_test400_holdout_v1"]:
        val.update("val-"+im["source_image_id"] for im in read_json(SHARED/name/"image_manifest.json")["images"])
    exposures=read_json(REPO/"docs/handoffs/data_eval/language_targeted_20261117_evidence/dev_exposure_registry.json")["images"]
    val.update("val-"+str(im["image_id"]-10**9-1) for im in exposures)
    bridges=[c for c in sorted(components) if parent.intersection(c) and val.intersection(c)]
    write_json(RUN/"frozen_bridge_clues.json",{"review_status":"unconfirmed hash/URL group connections; related new candidates quarantined","components":bridges,"count":len(bridges),"quarantined_new_source_images":sum(len(set(c)-frozen) for c in bridges)})

def run():
    ledger=read_json(RUN/"ledger.json")
    record_bridges(ledger)
    old=read_json(FULL/"old_subset_membership.json")
    new=read_json(FULL/"added_subset_membership.json")
    final_ids={i["image_id"] for i in read_json(FULL/"image_manifest.json")["images"]}
    old_ids=set(old["image_ids"]);new_ids=set(new["image_ids"])
    require(not old_ids&new_ids and final_ids==old_ids|new_ids,"membership union")
    require(new_ids=={int(k)+1 for k,v in ledger.items() if v=="added"},"ledger additions")
    require(old_ids=={int(k)+1 for k,v in ledger.items() if v=="parent"},"ledger parent")
    reasons=Counter(ledger.values())
    require(len(ledger)==45623 and not reasons["pending"],"full source coverage")
    reports={subset:report(root) for subset,root in [("old10k",PARENT),("added",DELTA),("full",FULL)]}
    baseline=read_json(PARENT/"audit/distribution_report.json")["combined"]
    for metric,field in [("small","small_instances_le1024"),("extreme","extreme_instances_le256")]:
        require(reports["old10k"]["overall"][metric]["instances"]==baseline[field],"parent baseline")
    require(reports["full"]["overall"]["all"]["groups"]==9976+len(new_ids),"new independent groups")
    geom=read_json(RUN/"topology_rows.json")
    labels={}
    for label,match in [("multiple_foreground_components",lambda n,h:n>1),("interior_holes",lambda n,h:h>0)]:
        for cid in (None,23,24):
            hits=[r for r in geom if (cid is None or r[2]==cid) and match(r[3],r[4])]
            labels[f"{label}_{cid}"]={"images":len({r[0] for r in hits}),"groups":len({r[0] for r in hits}),"instances":len(hits)}
    prior_visual=SHARED/"fp_train20000_targeted_v1/audit/visual_audit.json"
    visual=[r for r in read_json(prior_visual)["observations"] if r["image_id"] in new_ids]
    case_ids={"shoe_thin_straps":[41296],"shoe_open_toe_visible_gap":[148228],"bag_shoulder_strap":[307809,331715],"separated_visible_shoe_regions":[146511]}
    included={r["source_annotation_id"]:r for r in visual}
    observations={k:[included[i] for i in ids if i in included] for k,ids in case_ids.items()}
    boundary={"source_cutout":"See coverage source_cutout_all/shoe/bag; attribute 308","new_geometry":labels,"geometry_basis":"OpenCV RETR_CCOMP on unscaled native foreground; columns image id/project ann id/source category/foreground components/holes. Older 10k pixel audit reused; old topology unknown.","prior_AI_review_sha256":sha256(prior_visual),"prior_AI_images_inspected":36,"retained_AI_images":len(visual),"retained_AI_shoe_images":sum(r["source_category"]=="shoe" for r in visual),"retained_AI_bag_images":sum(r["source_category"]=="bag, wallet" for r in visual),"AI_example_hits":observations,"human_checks_this_run":0,"interpretation":"Documented visible evidence only. Open toe does not establish cutout manufacture; multiple components do not establish occlusion. Remaining semantic statuses unknown. Prior 10k qualitative observations in referenced parent visual audit."}
    boundary["new_targeted_AI_review"]=read_json(RUN/"visual_observations.json")
    write_json(RUN/"coverage.json",reports);write_json(RUN/"boundary.json",boundary)
    for path,digest in PINS.items():require(sha256(PARENT/path)==digest,"parent changed")
    for p,h in read_json(STAGED/"protection_inputs.json").items():require(sha256(p)==h,"protected input changed")
    audit=FULL/"audit";audit.mkdir(exist_ok=True)
    from PIL import Image
    modes=Counter()
    for im in read_json(DELTA/"image_manifest.json")["images"]:
        with Image.open(DELTA/im["file_name"]) as image:
            modes[image.mode]+=1
            require(image.size==(im["width"],im["height"]),"original size")
            if image.mode!="RGB":
                rgb=image.convert("RGB")
                rgb.load()
                require(rgb.mode=="RGB" and rgb.size==image.size,"RGB conversion")
    write_json(RUN/"image_modes.json",{"original_modes":dict(modes),"non_rgb_verified_with_loader_conversion":sum(v for k,v in modes.items() if k!="RGB")})
    for f in ["image_modes.json","ledger.json","counts.json","frozen_bridge_clues.json","duplicates.json","coverage.json","boundary.json","source_fidelity.json","visual_sample.json","visual_observations.json","shoe_hole.jpg","bag_sparse.jpg","separated.jpg","topology_rows.json","download_provenance.json","eligible.json"]:
        shutil.copyfile(RUN/f,audit/f)
    for f in ["protection_inputs.json","protected_fingerprints.json","input_preflight.json"]:shutil.copyfile(STAGED/f,audit/f)
    shutil.copyfile(PARENT/"audit/visual_audit.json",audit/"parent_visual_audit.json")
    release=dict(schema_version=3,illustrative_only=False,release_id=NAME,parent=PARENT.name,parent_pins=PINS,request_commit="dd786c74622820ef59ce2f5edc771453c3f25e94",build_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),source_annotation_sha256=sha256(SOURCE),mapping_sha256=sha256(FULL/"mapping_config.json"),source_account=dict(reasons),images=len(final_ids),groups=9976+len(new_ids),instances=len(old["annotation_ids"])+len(new["annotation_ids"]),added_images=len(new_ids),added_groups=len(new_ids),added_instances=len(new["annotation_ids"]),added_cpu_view=str(DELTA),source_fidelity_sha256=sha256(audit/"source_fidelity.json"),dev_exposure_registry_sha256=sha256(REPO/"docs/handoffs/data_eval/language_targeted_20261117_evidence/dev_exposure_registry.json"),consumer_acceptance="pending_lcy",formal_freeze=False,training_performed=False,python=platform.python_version())
    write_json(FULL/"RELEASE.json",release)
    entries=read_json(RUN/"asset_hashes.json")
    for f in FULL.rglob("*"):
        rel=f.relative_to(FULL).as_posix()
        if f.is_file() and rel not in entries and rel!="SHA256SUMS.json":entries[rel]=sha256(f)
    write_json(FULL/"SHA256SUMS.json",entries)
    final=SHARED/NAME;require(not final.exists(),"immutable version already exists")
    FULL.rename(final)
    write_json(RUN/"published.json",{**release,"path":str(final),"manifest_sha256":sha256(final/"image_manifest.json"),"annotations_sha256":sha256(final/"annotations_coco.json"),"release_sha256":sha256(final/"RELEASE.json"),"checksums_sha256":sha256(final/"SHA256SUMS.json")})
    print("published",final,release["images"],flush=True)

if __name__=="__main__":run()
