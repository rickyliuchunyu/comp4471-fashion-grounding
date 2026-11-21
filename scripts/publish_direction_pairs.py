"""Publish two-target bilingual spatial supervision from visually retained sheet entries."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from prepare_direction_pairs import P, read_json, write_json
from publish_language_expansion import build

NOUNS = {
    13: ("眼镜", "pair of glasses"), 15: ("发饰", "hair ornament"), 17: ("手套", "glove"),
    20: ("护腿套", "leg warmer"), 22: ("袜子", "sock"), 23: ("鞋", "shoe"),
    24: ("包", "bag"), 26: ("雨伞", "umbrella"), 101: ("上衣", "top"), 104: ("外套", "jacket"),
}


def caption(noun, axis, side, style):
    cn, en = noun
    short, position, english, relative = (
        ("左", "左侧", "left", "farther left"), ("右", "右侧", "right", "farther right"),
        ("上", "上方", "upper", "higher up"), ("下", "下方", "lower", "lower down"),
    )[axis * 2 + side]
    zh = (f"画面{position}的{cn}", f"图中位置更靠{short}的{cn}", f"两者中偏{short}的{cn}")[style]
    if axis == 0:
        en_text = (f"the {en} on the {english} in the picture",
                   f"the {en} {relative} in the image", f"the {en} on the {english} of the two")[style]
    else:
        en_text = (f"the {english} {en} in the picture", f"the {en} {relative} in the image",
                   f"the {english} {en} of the two")[style]
    return {"en": en_text, "zh": zh}


def publish(run, split):
    keep = read_json(run/"visual_keeps.json")
    chosen = []
    review_log = []
    package_id = "fp_language_direction_pairs_" + split + "_v1"
    for batch in (split, split+"extra"):
        sheets = read_json(run/(batch+"-proposals.json"))
        indices = set(keep[batch])
        assert indices <= {s["index"] for s in sheets}
        for s in sheets:
            keep_pair = s["index"] in indices
            review_log.append({**s, "batch": batch, "retained": keep_pair})
            if keep_pair:
                chosen.append(s)
    assert len({s["image_id"] for s in chosen}) == len(chosen)
    write_json(run/(split+"-visual-screen.json"),review_log)
    selected, decisions = [], {}
    for s in chosen:
        assert len(set(s["ids"]))==2 and s["n"]==2
        assert split != "train" or s["fine"]!=23
        pair_id = f"{package_id}-image-{s['image_id']}"
        for side, aid in enumerate(s["ids"]):
            idx = len(selected)+1
            selected.append(dict(image_id=s["image_id"],index=idx,target_id=aid,sheet=s["sheet"],contrast_set_id=pair_id))
            decisions[str(idx)] = dict(texts=caption(NOUNS[s["fine"]],s["axis"],side,s["image_id"]%3),
                type="spatial", reason="Original frame and both mask crops checked; two visibly separate items, ordered in image coordinates.")
    source = P/"fp_train_full_eligible_v1" if split=="train" else run/"dev_source"
    base = P/"fp_language1149_targeted_draft_v1" if split=="train" else None
    output = run/"staging"/package_id
    report = build(source,selected,decisions,output,split,base,annotator="codex_visual_author_20261118",
                   template_family="bilingual_opposite_image_direction_v1")
    import json
    rows=[json.loads(s) for s in (output/"queries_increment.jsonl").read_text().splitlines()]
    grouped=defaultdict(list)
    for q in rows:
        grouped[q["contrast_set_id"]].append(q)
    assert len(grouped)==len(chosen)
    for s, pair in zip(chosen,grouped.values(),strict=True):
        assert len(pair)==4 and {q["target_instance_id"] for q in pair}==set(s["ids"])
        assert len({q["query"] for q in pair})==4
        assert len({q["image_id"] for q in pair})==1
        assert all(q["review_status"]=="needs_revision" and q["reviewer"] is None for q in pair)
        assert all({q["language"] for q in pair if q["target_instance_id"]==aid}=={"zh","en"} for aid in s["ids"])
    if base:
        assert (output/"queries_machine_train.jsonl").read_bytes()==(base/"queries_machine_train.jsonl").read_bytes()+(output/"queries_increment.jsonl").read_bytes()
    else:
        assert (output/"queries_machine_train.jsonl").stat().st_size==0
    assert (output/"queries_accepted.jsonl").stat().st_size==0
    report.update(pairs=len(chosen), target_nouns=dict(Counter(NOUNS[s["fine"]][1] for s in chosen)),
                  axes=dict(Counter(("horizontal","vertical")[s["axis"]] for s in chosen)),
                  inspected=len(review_log), discarded=len(review_log)-len(chosen))
    write_json(run/(split+"-pair-report.json"),report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("split",choices=("train","dev"))
    args=p.parse_args()
    publish(args.run,args.split)
