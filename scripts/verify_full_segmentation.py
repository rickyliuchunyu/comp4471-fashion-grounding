"""Check the published metadata, source coverage, isolation and CPU-view provenance."""
from collections import Counter
from full_train_inventory import RUN, NAME, PARENT, SHARED, STAGED, read_json, require, sha256, write_json
from build_full_segmentation import DELTA
from build_staged_segmentation import verify_membership

def verify():
    root=SHARED/NAME
    meta=read_json(RUN/"published.json")
    catalog=read_json(root/"SHA256SUMS.json")
    require(sha256(root/"SHA256SUMS.json")==meta["checksums_sha256"],"published checksums")
    for path,digest in catalog.items():
        if path.startswith(("images/","masks/")):continue
        require(sha256(root/path)==digest,"published evidence differs: "+path)
    verify_membership(PARENT,DELTA,root)
    raw=read_json(DELTA/"source_subset.json")
    rules={r["source_id"]:r for r in read_json(root/"mapping_config.json")["fashionpedia"]}
    legal={a["id"]:a for a in raw["annotations"] if rules[a["category_id"]]["action"]=="map"}
    anns=read_json(DELTA/"annotations_coco.json")["annotations"]
    require(len(anns)==len(legal) and not any(a["category_id"]==11 for a in raw["annotations"]),"source coverage")
    require({a["source_annotation_id"] for a in anns}==set(legal),"missing source annotations")
    for a in anns:
        s=legal[a["source_annotation_id"]]
        require((a["id"],a["image_id"],a["category_id"],a["source_category_id"],a["iscrowd"],a["attribute_ids"])==(s["id"]+1,s["image_id"]+1,rules[s["category_id"]]["project_category_id"],s["category_id"],s["iscrowd"],s.get("attribute_ids",[])),"source identity")
    members=read_json(root/"audit/eligible.json")
    frozen=read_json(root/"audit/protected_fingerprints.json")
    from build_large_packages import specific_source_url
    import numpy as np
    full=frozen+members
    orig=np.array([r["phash"] for r in full],np.uint64)
    flip=np.array([r["phash_flip"] for r in full],np.uint64)
    hashes={r["sha256"] for r in frozen};urls={specific_source_url(r.get("original_url")) for r in frozen}
    for i in range(len(frozen),len(full)):
        r=full[i];u=specific_source_url(r.get("original_url"))
        require(r["sha256"] not in hashes and (u is None or u not in urls),"SHA/URL collision")
        hashes.add(r["sha256"]);urls.add(u)
        for h in [orig[i],flip[i]]:
            require(not np.any(np.bitwise_count(orig[:i]^h)<=6) and not np.any(np.bitwise_count(flip[:i]^h)<=6),"retained near-duplicate")
    report={"status":"passed","retained_new_images":len(members),"all_new_legal_instances":len(legal),"protected_fingerprints":len(frozen),"retained_duplicate_flags":0,"parent_and_delta_union_verified":True,"checksums_sha256":meta["checksums_sha256"]}
    write_json(RUN/"postpublish_check.json",report)
    print(report)
if __name__=="__main__":verify()
