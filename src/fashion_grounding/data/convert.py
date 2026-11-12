"""Fashionpedia training pilot -> COCO, schema-3 GT candidates and draft queries."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from PIL import Image

from .schema import decode_mask, encode_mask, mask_box, read_json, require, sha256, write_json

COMMON = {"schema_version": 3, "illustrative_only": False}
NAMES = {1: ("上衣", "top"), 2: ("裤子", "pair of trousers"), 3: ("裙子", "skirt"), 4: ("外套", "outerwear garment"), 5: ("连衣裙", "dress"), 6: ("鞋子", "shoe"), 7: ("包", "bag"), 8: ("配饰", "accessory")}


def convert_image(image, annotations, mapping, image_path):
    """All-or-nothing per image. Preserve crowd in COCO, exclude it from GT candidates."""
    if any(mapping[a["category_id"]]["action"] == "exclude_image" for a in annotations):
        raise ValueError("exclude_image:jumpsuit")
    with Image.open(image_path) as decoded:
        decoded.load()
        require(decoded.size == (image["width"], image["height"]), "source image size mismatch")
    output, masks = [], {}
    for a in annotations:
        rule = mapping[a["category_id"]]
        if rule["action"] == "exclude_annotation":
            continue
        require(rule["action"] == "map", "unknown mapping action")
        mask = decode_mask(a["segmentation"], image["height"], image["width"])
        box = mask_box(mask)
        # Official IDs may include zero. Reserve 1..1e9 for official training.
        aid, iid = int(a["id"]) + 1, int(image["id"]) + 1
        require(0 < aid < 10**9 and 0 < iid < 10**9, "source ID outside reserved range")
        row = {"id": aid, "image_id": iid, "category_id": rule["project_category_id"], "segmentation": encode_mask(mask), "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]], "area": int(mask.sum()), "iscrowd": a["iscrowd"], "source_annotation_id": a["id"], "source_category_id": a["category_id"], "source_category_name": rule["source_name"], "source_dataset": "fashionpedia", "attribute_ids": a.get("attribute_ids", [])}
        output.append(row)
        masks[aid] = mask
    require(bool(output), "no whole instances")
    return sorted(output, key=lambda r: r["id"]), masks


def draft_queries(image, annotations, target_counts=None):
    target_counts = target_counts if target_counts is not None else Counter()
    valid = [a for a in annotations if not a["iscrowd"]]
    by_category = defaultdict(list)
    for a in valid:
        by_category[a["category_id"]].append(a)
    targets = []
    # Prefer genuine spatial competition; do not fabricate colors or materials.
    for cat in (7, 1, 4, 5, 3, 2, 6, 8):
        group = sorted(by_category[cat], key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
        if len(group) < 2:
            continue
        if target_counts[cat] >= 4:
            continue
        if cat == 8 and len({a["source_category_id"] for a in group}) != 1:
            continue
        centers = [a["bbox"][0] + a["bbox"][2] / 2 for a in group]
        if min(centers[1]-centers[0], centers[-1]-centers[-2]) > image["width"] * 0.03:
            targets = [(group[0], "leftmost"), (group[-1], "rightmost")]
            break
    if not targets:
        unique = [(group[0], None) for cat, group in by_category.items() if len(group) == 1]
        targets = sorted(unique, key=lambda pair: (target_counts[pair[0]["category_id"]], pair[0]["category_id"]))[:2]
    rows = []
    for a, direction in targets:
        target_counts[a["category_id"]] += 1
        zh, en = NAMES[a["category_id"]]
        semantic = f"pilot-train-{image['image_id']}-ann-{a['id']}"
        contrast = f"pilot-spatial-{image['image_id']}-{a['category_id']}" if direction else None
        texts = {"zh": ("最左边的" if direction == "leftmost" else "最右边的") + zh if direction else "图中的" + zh, "en": f"the {direction} {en}" if direction else f"the {en} in the image"}
        for language, text in texts.items():
            rows.append({**COMMON, "query_id": semantic + "-" + language, "semantic_query_id": semantic, "contrast_set_id": contrast, "image_id": image["image_id"], "source_dataset": "fashionpedia", "split": "train", "query": text, "language": language, "target_instance_id": a["id"], "target_category_id": a["category_id"], "same_category_instance_count": len(by_category[a["category_id"]]), "query_types": ["category", "spatial"] if direction else ["category"], "spatial_reference": "image_view", "annotation_version": "pilot-draft-v1", "annotator": "codex_template_generator", "reviewer": None, "review_status": "needs_revision", "target_source_category_id": a["source_category_id"], "target_source_category_name": a["source_category_name"], "same_source_category_instance_count": sum(x["source_category_id"] == a["source_category_id"] for x in valid), "authorship": "machine_generated_template_pending_human_review", "template_family": "spatial_extreme_v1" if direction else "unique_category_v1", "paraphrase_group_id": None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    require(not args.output.exists(), "output already exists; publish a new version")
    source_path = args.raw / "instances_attributes_train2020.json"
    selection = read_json(args.raw / "pilot_selection.json")
    require(selection["source_split"] == "official_train2020", "pilot must use official training")
    require(sha256(source_path) == selection["source_annotation_sha256"], "source changed")
    source = read_json(source_path)
    config_path = repo / "configs/source_category_mapping.json"
    config = read_json(config_path)
    mapping = {r["source_id"]: r for r in config["fashionpedia"]}
    require({c["id"]: c["name"] for c in source["categories"]} == {k: v["source_name"] for k, v in mapping.items()}, "source category names differ from configured mapping")
    selected = set(selection["image_ids"])
    source_images = sorted([im for im in source["images"] if im["id"] in selected], key=lambda r: r["id"])
    require(len(source_images) == len(selected), "selected IDs missing/duplicated")
    grouped = defaultdict(list)
    for a in source["annotations"]:
        if a["image_id"] in selected:
            grouped[a["image_id"]].append(a)
    out = args.output
    out.mkdir(parents=True)
    issues, manifest_images, coco_images, converted, candidate_images, queries = [], [], [], [], [], []
    target_counts = Counter()
    for im in source_images:
        raw_image = args.raw / "images" / Path(im["file_name"]).name
        try:
            anns, masks = convert_image(im, grouped[im["id"]], mapping, raw_image)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            issues.append({"source_image_id": im["id"], "error": str(exc), "decision": "quarantine_whole_image"})
            continue
        filename = "images/" + raw_image.name
        (out / "images").mkdir(exist_ok=True)
        shutil.copyfile(raw_image, out / filename)
        iid = im["id"] + 1
        row = {"image_id": iid, "file_name": filename, "sha256": sha256(out / filename), "width": im["width"], "height": im["height"], "source_dataset": "fashionpedia", "source_image_id": str(im["id"]), "source_split": "official_train2020", "group_id": "fp-sha256-" + sha256(out / filename), "split": "train"}
        manifest_images.append(row)
        coco_images.append({**{k: v for k, v in row.items() if k != "image_id"}, "id": iid, "license": im["license"], "supervised_category_ids": list(range(1, 9))})
        converted.extend(anns)
        cs = []
        for a in anns:
            if a["iscrowd"]:
                continue
            reference = f"masks/{a['id']}.json"
            write_json(out / reference, a["segmentation"])
            cs.append({"candidate_id": f"fp-train-gt-{a['id']}", "category_id": a["category_id"], "score": None, "bbox_xyxy": mask_box(masks[a["id"]]), "mask_reference": reference, "mask_sha256": sha256(out / reference), "decoder_query_id": None, "source_annotation_id": a["id"]})
        candidate_images.append({"image_id": iid, "image_size_wh": [im["width"], im["height"]], "status": "ok" if cs else "no_candidates", "candidates": cs})
        queries.extend(draft_queries(row, anns, target_counts))
    (out / "conversion_issues.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in issues), encoding="utf-8")
    require(not issues and len(manifest_images) == selection["count"], "quarantined images; inspect conversion_issues and choose a new version")
    cats = read_json(repo / "configs/categories.json")["categories"]
    write_json(out / "annotations_coco.json", {**COMMON, "info": {"description": "Training-only interface pilot; not a benchmark", "source_annotation_sha256": sha256(source_path)}, "licenses": source["licenses"], "images": coco_images, "annotations": converted, "categories": cats})
    write_json(out / "image_manifest.json", {**COMMON, "manifest_id": out.name, "mapping_version": config["mapping_version"], "images": manifest_images})
    write_json(out / "candidates_gt.json", {**COMMON, "candidate_set_id": out.name + "-gt", "candidate_source": "gt", "image_manifest": {"path": "image_manifest.json", "sha256": sha256(out / "image_manifest.json")}, "provenance": {"producer_run_id": "wkq-pilot-conversion-v1", "checkpoint_sha256": None, "decoder_config_sha256": None, "annotation_manifest_sha256": sha256(out / "annotations_coco.json")}, "images": candidate_images})
    (out / "queries_draft.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    (out / "queries_accepted.jsonl").write_text("", encoding="utf-8")
    # Preserve the source subset and licensing/provenance so mask conversion is independently auditable.
    write_json(out / "source_subset.json", {"info": source["info"], "licenses": source["licenses"], "categories": source["categories"], "images": source_images, "annotations": [a for im in source_images for a in grouped[im["id"]]]})
    shutil.copyfile(config_path, out / "mapping_config.json")
    shutil.copyfile(args.raw / "pilot_selection.json", out / "selection.json")
    shutil.copyfile(args.raw / "download_provenance.json", out / "download_provenance.json")
    stats = {**COMMON, "status": "converted_pending_human_review", "images": len(manifest_images), "source_annotations": sum(len(x) for x in grouped.values()), "whole_annotations": len(converted), "excluded_parts": sum(mapping[a["category_id"]]["action"] == "exclude_annotation" for group in grouped.values() for a in group), "crowd_annotations": sum(bool(a["iscrowd"]) for a in converted), "instances_per_category": dict(Counter(a["category_id"] for a in converted)), "source_categories": dict(Counter(a["source_category_name"] for a in converted)), "area_bins": dict(Counter("small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large" for a in converted)), "draft_query_records": len(queries), "semantic_queries": len({q["semantic_query_id"] for q in queries}), "accepted_query_records": 0, "exact_duplicate_groups": len(manifest_images) - len({im["sha256"] for im in manifest_images}), "near_duplicate_review": "not_done; all selected images are train; no dev/test split published", "mapping_config_sha256": sha256(config_path), "id_policy": "official train image/annotation ID + 1; reserve >=1e9 for future source/split namespaces", "human_mask_review": "pending", "consumer_acceptance": "pending"}
    write_json(out / "conversion_report.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
