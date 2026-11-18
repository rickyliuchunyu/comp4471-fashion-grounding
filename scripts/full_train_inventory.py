"""Inventory and de-duplicate all remaining Fashionpedia train images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from expand_segmentation_staged import (
    B, REPO, PARENT, PRIOR, OLD_RUN, SHARED, RUN as STAGED, SOURCE, PINS,
    sources, incremental_groups, read_json, write_json, require, sha256, phash,
)
from acquire_large_pool import download_members

RUN = B / "runs/data_eval/20261118-full-train-v1"
URL = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
NAME = "fp_train_full_eligible_v1"

def prepare(step):
    release = read_json(SHARED/"fp_train20000_targeted_v1/SHA256SUMS.json")
    for path, digest in PINS.items():
        require(sha256(PARENT/path) == digest, "10k parent changed")
    for name in ("protected_fingerprints.json", "protection_inputs.json"):
        require(sha256(STAGED/name) == release["audit/"+name], "staged protection changed")
    protection = read_json(STAGED/"protection_inputs.json")
    require(all(sha256(p)==v for p,v in protection.items()), "protected inputs changed")
    require(all(str(p) in protection for p in SHARED.glob("fp_language_dev*/image_manifest.json")), "additional dev protection")
    require(sha256(SOURCE)==read_json(PARENT/"RELEASE.json")["source_annotation_sha256"], "official source changed")
    data, by = sources()
    old = {int(im["source_image_id"]) for im in read_json(PARENT/"image_manifest.json")["images"]}
    protected = read_json(STAGED/"protected_fingerprints.json")
    frozen = {x["key"] for x in protected}
    cache = {}
    for file in [PRIOR/"fingerprints.json", OLD_RUN/"new_fingerprints.json", STAGED/"pool_fingerprints.json"]:
        for x in read_json(file):
            if x["key"] in cache and not x.get("group_id"):
                x["group_id"] = cache[x["key"]].get("group_id")
            cache[x["key"]] = x
    stats = read_json(STAGED/"metadata_stats.json")
    invalid = {e["source_image_id"] for e in read_json(STAGED/"feasibility_metadata.json")["errors"]}
    ledger = {}
    raw = RUN/"raw"
    raw.mkdir(exist_ok=True)
    images = []
    missing = []
    for im in data["images"]:
        i = im["id"]; key = f"train-{i}"
        cats = {a["category_id"] for a in by[i]}
        if i in old: status = "parent"
        elif 11 in cats: status = "excluded_jumpsuit"
        elif not any(c < 27 for c in cats): status = "excluded_no_whole"
        elif i in invalid: status = "excluded_invalid_source_mask"
        elif key in frozen: status = "isolated_reserved_group"
        else:
            require(str(i) in stats, "unaccounted source image")
            status = "pending"; images.append(im)
            if key not in cache or not Path(cache[key]["path"]).is_file(): missing.append(im)
        ledger[str(i)] = status
    require(len(ledger)==len(data["images"]), "source id collision")
    if step == "download":
        print("new downloads",len(missing),flush=True)
        write_json(RUN/"download_provenance.json",download_members(URL,missing,raw,workers=40))
        return
    def fingerprint(im):
        key = "train-"+str(im["id"])
        prev = cache.get(key)
        p = Path(prev["path"]) if prev and Path(prev["path"]).is_file() else raw/Path(im["file_name"]).name
        digest = sha256(p)
        if prev:
            require(prev["sha256"]==digest, "cached image changed "+key)
            return {**prev,"path":str(p),"source_image":im}
        with Image.open(p) as image:
            image.load()
            require(image.size==(im["width"],im["height"]), "image size "+key)
            return dict(key=key,source_image=im,path=str(p),source_split="train",sha256=digest,original_url=im.get("original_url"),phash=phash(image),phash_flip=phash(ImageOps.mirror(image)),group_id=None)
    with ThreadPoolExecutor(max_workers=5) as pool: items = list(pool.map(fingerprint,images))
    keys, assignment, edges, excluded = incremental_groups(protected,items)
    index = {r["key"]:r for r in items}
    for r in excluded: ledger[r["key"][6:]] = r["reason"]
    for k in keys: ledger[k[6:]] = "added"
    require("pending" not in ledger.values(), "incomplete full source")
    write_json(RUN/"eligible.json",[{**index[k],"group_id":"fp-full-"+k} for k in sorted(keys)])
    write_json(RUN/"ledger.json",ledger)
    write_json(RUN/"duplicates.json",{"edges":edges,"excluded":excluded,"protected_count":len(protected),"policy":"incremental four-orientation pHash <=6, resource URL, group and SHA; one lexicographic representative, quarantine all protected components"})
    write_json(RUN/"counts.json",dict(Counter(ledger.values())))
    print(Counter(ledger.values()),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("step",choices=["download","select"])
    prepare(parser.parse_args().step)
