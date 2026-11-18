"""Prepare protected official-validation assets and image/mask sheets for opposite-direction pairs."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from build_large_packages import build_package
from expand_segmentation_staged import incremental_groups
from fashion_grounding.data.schema import read_json, write_json, decode_mask

B = Path("/root/autodl-tmp/fashion-grounding")
P = B / "shared/processed"
OLD = B / "runs/data_eval/20261112-large-data-regrouped"
PREV = B / "runs/data_eval/20261117-language-targeted-expansion-r02"


def dev_source(run):
    audit = P / "fp_train_full_eligible_v1/audit"
    protected = read_json(audit / "protected_fingerprints.json") + read_json(audit / "eligible.json")
    blocked = {x["key"] for x in protected}
    inventory = {x["key"]: x for x in read_json(OLD / "fingerprints.json")}
    unused = read_json(OLD / "split_plan.json")["validation_unused"]
    keys, _, edges, conflicts = incremental_groups(protected, [inventory[k] for k in sorted(unused) if k not in blocked])
    write_json(run / "dev_pool_guard.json", dict(eligible=keys, collisions=conflicts, edges=edges,
                                                protected=len(protected), prior_unused=len(unused)))
    sel = read_json(OLD / "raw/selection.json")["selections"]["val"]
    data = read_json(sel["source_path"])
    groups = defaultdict(list)
    for a in data["annotations"]:
        groups[a["image_id"]].append(a)
    cfg = read_json(Path("configs/source_category_mapping.json"))
    mapping = {a["source_id"]: a for a in cfg["fashionpedia"]}
    cats = read_json(Path("configs/categories.json"))["categories"]
    sources = {"val": dict(data=data, by=groups, hash=sel["source_sha256"])}
    build_package(run / "dev_source", "dev", sorted(keys), inventory, sources, mapping, cfg, cats)


def options(anns, im, split, relaxed=False):
    buckets = defaultdict(list)
    for a in anns:
        buckets[a["source_category_id"]].append(a)
    choices = []
    for fine, items in buckets.items():
        if len(items) < 2 or (split == "train" and fine == 23) or any(a["iscrowd"] for a in items):
            continue
        for axis, extent in enumerate((im["width"], im["height"])):
            center = lambda a: a["bbox"][axis] + a["bbox"][axis+2]/2
            ranked = sorted(items, key=center)
            lo, hi = ranked[0], ranked[-1]
            if min(lo["area"], hi["area"]) < (100 if relaxed else 270) or min(*lo["bbox"][2:], *hi["bbox"][2:]) < (5 if relaxed else 8):
                continue
            margin = min(center(ranked[1])-center(lo), center(hi)-center(ranked[-2])) / extent
            separation = (center(hi)-center(lo)) / max(lo["bbox"][axis+2],hi["bbox"][axis+2])
            if margin < (.038 if relaxed else .075) or separation < (.55 if relaxed else .78):
                continue
            quality = min(margin,.55) + .13*min(separation,2) + .04*np.log1p(min(hi["area"],lo["area"])/300)
            choices.append(dict(image_id=im["image_id"], fine=fine, name=lo["source_category_name"],
                                axis=axis, ids=[lo["id"],hi["id"]], n=len(items), quality=float(quality)))
    return choices


def shortlist(source, split, run):
    manifest = read_json(source / "image_manifest.json")["images"]
    excluded = set()
    if split == "train":
        prior = read_json(P / "fp_language1149_targeted_draft_v1/image_manifest.json")
        excluded = {i["group_id"] for i in prior["images"]}
        excluded_ids = {x["image_id"] for x in read_json(PREV / "train-selection-complete.json")}
        clues = read_json(P / "fp_train_full_eligible_v1/audit/frozen_bridge_clues.json")["components"]
        excluded_ids.update(int(k.split("-")[1])+1 for c in clues for k in c if k.startswith("train-"))
        manifest = [im for im in manifest if im["image_id"] not in excluded_ids]
    by = defaultdict(list)
    all_anns = read_json(source / "annotations_coco.json")["annotations"]
    for a in all_anns:
        by[a["image_id"]].append(a)
    ims = {im["image_id"]: im for im in manifest}
    ann = {a["id"]: a for a in all_anns}
    selection = [s for im in manifest if im["group_id"] not in excluded for s in options(by[im["image_id"]],im,split)]
    # Round-robin over source nouns; reserve shoe slots for the known dev failure mode only.
    buckets = defaultdict(list)
    for s in sorted(selection, key=lambda s:(-s["quality"],s["image_id"])):
        buckets[s["fine"]].append(s)
    picked, used = [], set()
    for round_index in range(150):
        for fine, bucket in sorted(buckets.items()):
            limit = (68 if split=="dev" else 0) if fine==23 else (24 if split=="train" else 40)
            if round_index>=limit:
                continue
            while bucket:
                s = bucket.pop(0)
                gid = ims[s["image_id"]]["group_id"]
                if gid in used:
                    continue
                picked.append(s)
                used.add(gid)
                break
        if split=="train" and len(picked)>=152:
            break
    picked.sort(key=lambda s:(s["fine"],s["image_id"]))
    render(picked, ims, ann, source, split, run)


def render(picked, ims, ann, source, split, run):
    for i,s in enumerate(picked):
        s["index"] = i+1
        s["sheet"] = f"{split}-{i//12+1:02}.jpg"
    write_json(run/(split+"-proposals.json"),picked)
    print(split, len(picked), Counter(s["fine"] for s in picked),flush=True)
    for start in range(0,len(picked),12):
        canvas = Image.new("RGB",(1200,800),"white")
        draw = ImageDraw.Draw(canvas)
        for j,s in enumerate(picked[start:start+12]):
            im=ims[s["image_id"]]
            with Image.open(source/im["file_name"]) as inp:
                pic=inp.convert("RGB")
            x=(j%3)*400
            y=(j//3)*200
            draw.text((x+2,y+2),f"{s['index']} {s['name']} {'L/R' if s['axis']==0 else 'U/D'} n={s['n']}",fill="black")
            thumb=ImageOps.contain(pic,(178,175))
            canvas.paste(thumb,(x,y+19))
            for k,aid in enumerate(s["ids"]):
                a=ann[aid]
                bx,by0,bw,bh=a["bbox"]
                col=(0,210,240) if k==0 else (255,85,20)
                rect=[x+v*thumb.width/pic.width if t%2==0 else y+19+v*thumb.height/pic.height for t,v in enumerate([bx,by0,bx+bw,by0+bh])]
                draw.rectangle(rect,outline=col,width=2)
                arr=np.array(pic)
                mask=decode_mask(a["segmentation"],pic.height,pic.width)
                arr[mask]=arr[mask]*.72+np.array(col)*.28
                crop=Image.fromarray(arr).crop((max(0,bx-6),max(0,by0-6),min(pic.width,bx+bw+6),min(pic.height,by0+bh+6)))
                canvas.paste(ImageOps.contain(crop,(105,166)),(x+180+k*108,y+19))
        canvas.save(run/picked[start]["sheet"],quality=90)


def supplement(source, split, run):
    prior = read_json(run / (split+"-proposals.json"))
    excluded = {s["image_id"] for s in prior}
    if split == "train":
        excluded.update(i["image_id"] for i in read_json(P/"fp_language1149_targeted_draft_v1/image_manifest.json")["images"])
        excluded.update(i["image_id"] for i in read_json(PREV/"train-selection-complete.json"))
        excluded.update(int(k.split("-")[1])+1 for c in read_json(P/"fp_train_full_eligible_v1/audit/frozen_bridge_clues.json")["components"] for k in c if k.startswith("train-"))
    manifest = read_json(source/"image_manifest.json")["images"]
    blocked_groups = {i["group_id"] for i in manifest if i["image_id"] in excluded}
    ims = {i["image_id"]:i for i in manifest if i["group_id"] not in blocked_groups}
    annotations = read_json(source/"annotations_coco.json")["annotations"]
    ann = {a["id"]:a for a in annotations}
    by = defaultdict(list)
    for a in annotations:
        if a["image_id"] in ims:
            by[a["image_id"]].append(a)
    pools = defaultdict(list)
    for iid, items in by.items():
        # Use the shared clothing noun only where both pieces are independently visible.
        expanded = [dict(a,source_category_id=100+a["category_id"],source_category_name="project-class-"+str(a["category_id"])) for a in items if a["category_id"] in (1,2,3,4,5)]
        for option in options(items+expanded,ims[iid],split,relaxed=(split=="dev")):
            fine = option["fine"]
            if fine not in (0,2,3,5,8,9,10,13,14,18,24,26,101,102,103,104,105) and not (split=="dev" and fine in (15,17,20,22)):
                continue
            if fine>=100 and len({ann[k]["source_category_id"] for k in option["ids"]})==1 and fine in (102,104):
                continue
            pools[fine].append(option)
    selected=[]
    for fine, items in sorted(pools.items(),key=lambda kv:(kv[0]!=24,kv[0])):
        limit= (48 if fine==24 else 12) if split=="train" else 30
        count=0
        for s in sorted(items,key=lambda q:-q["quality"]):
            g=ims[s["image_id"]]["group_id"]
            if g in blocked_groups:
                continue
            selected.append(s)
            blocked_groups.add(g)
            count+=1
            if count>=limit:
                break
    render(selected,ims,ann,source,split+"extra",run)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--supplement",action="store_true")
    args=p.parse_args()
    if args.supplement:
        supplement(args.run/"dev_source","dev",args.run)
        supplement(P/"fp_train_full_eligible_v1","train",args.run)
        raise SystemExit
    dev_source(args.run)
    shortlist(args.run/"dev_source","dev",args.run)
    shortlist(P/"fp_train10000_targeted_v1","train",args.run)
