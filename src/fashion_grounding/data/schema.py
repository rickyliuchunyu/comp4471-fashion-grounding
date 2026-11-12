"""CPU checks for real GT pilot packages; no model dependencies."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, value):
    require(isinstance(value, str) and value and ":" not in value and "\\" not in value, "unsafe path")
    rel = Path(value)
    require(not rel.is_absolute() and not PureWindowsPath(value).drive and ".." not in rel.parts, "unsafe path")
    target = (Path(root) / rel).resolve()
    require(target.is_relative_to(Path(root).resolve()), "path escapes package")
    return target


def decode_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        require(bool(segmentation), "empty polygon list")
        for polygon in segmentation:
            require(len(polygon) >= 6 and len(polygon) % 2 == 0, "invalid polygon")
            xy = np.asarray(polygon, dtype=float).reshape(-1, 2)
            require(np.isfinite(xy).all(), "non-finite polygon")
            require(((xy[:, 0] >= 0) & (xy[:, 0] <= width) & (xy[:, 1] >= 0) & (xy[:, 1] <= height)).all(), "polygon outside image")
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    else:
        require(segmentation["size"] == [height, width], "RLE size mismatch")
        if isinstance(segmentation["counts"], list):
            counts = segmentation["counts"]
            require(all(type(n) is int and n >= 0 for n in counts) and sum(counts) == height * width, "invalid RLE runs")
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
            rle["counts"] = rle["counts"].encode("ascii")
    mask = coco_mask.decode(rle)
    require(mask.shape == (height, width) and mask.any(), "empty mask or wrong dimensions")
    return mask.astype(bool)


def encode_mask(mask):
    rle = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def unique_index(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "duplicate " + key)
    return result


def validate_queries(rows, images, annotations, accepted_only=False):
    unique_index(rows, "query_id")
    semantics = {}
    for q in rows:
        require(q["schema_version"] == 3 and q["illustrative_only"] is False, "query schema")
        im, ann = images[q["image_id"]], annotations[q["target_instance_id"]]
        require(ann["image_id"] == q["image_id"] and not ann["iscrowd"], "invalid query target")
        require(q["split"] == im["split"] and q["target_category_id"] == ann["category_id"], "query split/category mismatch")
        require(q["language"] in ("zh", "en") and q["query"].strip(), "invalid query text")
        signature = (q["image_id"], q["target_instance_id"], q["split"])
        require(semantics.setdefault(q["semantic_query_id"], signature) == signature, "translation target mismatch")
        if accepted_only:
            require(q["review_status"] == "accepted", "unreviewed query cannot enter evaluation")
        if q["review_status"] == "accepted":
            require(q["reviewer"] and q["annotator"] and q["reviewer"] != q["annotator"], "independent review required")


def validate_package(root):
    root = Path(root)
    manifest = read_json(root / "image_manifest.json")
    package = read_json(root / "candidates_gt.json")
    coco = read_json(root / "annotations_coco.json")
    for obj in (manifest, package, coco):
        require(obj["schema_version"] == 3 and obj["illustrative_only"] is False, "real schema 3 required")
    require(package["candidate_source"] == "gt", "GT validator only")
    ref = package["image_manifest"]
    require(sha256(safe_path(root, ref["path"])) == ref["sha256"], "manifest hash mismatch")
    require(sha256(root / "annotations_coco.json") == package["provenance"]["annotation_manifest_sha256"], "annotation hash mismatch")
    images = unique_index(manifest["images"], "image_id")
    ci = unique_index(coco["images"], "id")
    annotations = unique_index(coco["annotations"], "id")
    require(set(images) == set(ci), "COCO image IDs mismatch")
    require(all(type(i) is int and i > 0 for i in [*images, *annotations]), "IDs must be positive integers")
    groups, hashes = {}, {}
    for iid, im in images.items():
        require(im["split"] in ("train", "dev", "test", "unused"), "invalid split")
        require(groups.setdefault(im["group_id"], im["split"]) == im["split"], "group leakage")
        require(hashes.setdefault(im["sha256"], im["split"]) == im["split"], "image hash leakage")
        path = safe_path(root, im["file_name"])
        require(sha256(path) == im["sha256"], "image hash mismatch")
        with Image.open(path) as image:
            image.load()
            require(image.size == (im["width"], im["height"]), "image size mismatch")
            image.convert("RGB")
        require(all(ci[iid][k] == im[k] for k in ("width", "height", "file_name")), "COCO image mismatch")
    for aid, ann in annotations.items():
        im = images[ann["image_id"]]
        require(ann["category_id"] in range(1, 9), "invalid category")
        mask = decode_mask(ann["segmentation"], im["height"], im["width"])
        box = mask_box(mask)
        require(ann["bbox"] == [box[0], box[1], box[2]-box[0], box[3]-box[1]], "COCO box mismatch")
        require(ann["area"] == int(mask.sum()), "COCO area mismatch")
        del mask  # Bound memory for large datasets; re-decode per candidate below.
    candidate_images = unique_index(package["images"], "image_id")
    require(set(candidate_images) == set(images), "candidate images incomplete")
    ids, refs = set(), set()
    for iid, row in candidate_images.items():
        im = images[iid]
        require(row["image_size_wh"] == [im["width"], im["height"]], "candidate image size mismatch")
        cs = row["candidates"]
        require(row["status"] == ("ok" if cs else "no_candidates"), "invalid candidate status")
        order = [c["source_annotation_id"] for c in cs]
        require(order == sorted(order) and len(set(order)) == len(order), "GT ordering/duplicates")
        for c in cs:
            aid = c["source_annotation_id"]
            ann = annotations[aid]
            require(c["candidate_id"] not in ids and bool(c["candidate_id"]), "candidate ID duplicate")
            ids.add(c["candidate_id"]); refs.add(aid)
            require(ann["image_id"] == iid and not ann["iscrowd"] and c["category_id"] == ann["category_id"], "candidate association mismatch")
            require(c["score"] is None and c["decoder_query_id"] is None, "GT score/query must be null")
            path = safe_path(root, c["mask_reference"])
            require(sha256(path) == c["mask_sha256"], "mask hash mismatch")
            mask = decode_mask(read_json(path), im["height"], im["width"])
            require(np.array_equal(mask, decode_mask(ann["segmentation"], im["height"], im["width"])), "candidate/COCO mask mismatch")
            require(c["bbox_xyxy"] == mask_box(mask), "candidate box mismatch")
    require(refs == {a for a, row in annotations.items() if not row["iscrowd"]}, "missing GT candidates")
    counts = {}
    for filename, strict in (("queries_draft.jsonl", False), ("queries_accepted.jsonl", True)):
        rows = [json.loads(line) for line in (root / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_queries(rows, images, annotations, accepted_only=strict)
        counts[filename] = len(rows)
    return {"status": "passed", "images": len(images), "instances": len(annotations), "candidates": len(ids), "queries": counts}
