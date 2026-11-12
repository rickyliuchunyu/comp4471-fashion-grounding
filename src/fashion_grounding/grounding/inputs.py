"""Grounding-side schema-3 consumer; reuses wkq's common readers and RLE tools."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from PIL import Image
from fashion_grounding.data.schema import read_json, sha256, safe_path, decode_mask, mask_box, require, unique_index

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    category_id: int
    score: float | None
    bbox_xyxy: tuple
    mask: np.ndarray
    decoder_query_id: int | None

@dataclass(frozen=True)
class Query:
    query_id: str
    image_id: int
    query: str
    language: str
    split: str
    illustrative_only: bool

def inference_query(row):
    """Explicit allowlist: target IDs, target category and query types never cross."""
    require(row["schema_version"] == 3, "query schema must be 3")
    require(isinstance(row["query_id"], str) and bool(row["query_id"]), "invalid query ID")
    require(isinstance(row["query"], str) and bool(row["query"].strip()), "empty text")
    require(row["language"] in ("zh", "en"), "unsupported language")
    require(type(row["image_id"]) is int and row["image_id"] > 0, "invalid image ID")
    require(type(row["illustrative_only"]) is bool, "illustrative flag required")
    return Query(**{k: row[k] for k in Query.__dataclass_fields__})

class CandidatePack:
    def __init__(self, path, allow_illustrative=False):
        self.path = Path(path).resolve()
        self.package = read_json(self.path)
        p = self.package
        require(p["schema_version"] == 3, "candidate schema must be 3")
        require(type(p["illustrative_only"]) is bool, "candidate illustrative flag required")
        require(allow_illustrative or not p["illustrative_only"], "synthetic pack requires opt-in")
        require(p["candidate_source"] in ("gt", "predicted"), "internal candidates required")
        require(isinstance(p["candidate_set_id"], str) and bool(p["candidate_set_id"]), "invalid set ID")
        self.digest = sha256(self.path)
        ref = p["image_manifest"]
        self.manifest_path = safe_path(self.path.parent, ref["path"])
        require(sha256(self.manifest_path) == ref["sha256"], "manifest hash mismatch")
        manifest = read_json(self.manifest_path)
        require(manifest["schema_version"] == 3, "image schema must be 3")
        require(manifest["illustrative_only"] == p["illustrative_only"], "mixed synthetic/real pack")
        self.images = unique_index(manifest["images"], "image_id")
        self.rows = unique_index(p["images"], "image_id")
        require(set(self.rows) <= set(self.images), "unknown candidate image")
        ids = set()
        for iid, row in self.rows.items():
            im = self.images[iid]
            require(row["image_size_wh"] == [im["width"], im["height"]], "image dimensions differ")
            cs, status = row["candidates"], row["status"]
            require(status in ("ok", "no_candidates", "inference_error"), "bad upstream status")
            require((status == "ok") == bool(cs), "status/candidate mismatch")
            if status == "inference_error":
                require(isinstance(row.get("error"), dict), "upstream failure lacks error")
            if p["candidate_source"] == "gt":
                order = [c["source_annotation_id"] for c in cs]
                require(all(type(a) is int and a > 0 for a in order), "bad GT reference")
                require(order == sorted(set(order)), "GT order/duplicate reference")
            else:
                require(all(isinstance(c["score"], (int,float)) and math.isfinite(c["score"]) for c in cs), "bad predicted score")
                require(cs == sorted(cs, key=lambda c: (-c["score"], c["candidate_id"])), "predicted candidate order")
            for c in cs:
                cid = c["candidate_id"]
                require(isinstance(cid, str) and cid and cid not in ids, "duplicate/invalid candidate ID")
                ids.add(cid)
                require(type(c["category_id"]) is int and c["category_id"] in range(1,9), "invalid category")
                if p["candidate_source"] == "gt":
                    require(c["score"] is None and c["decoder_query_id"] is None, "GT prediction fields")
                else:
                    require(c["source_annotation_id"] is None, "prediction contains GT answer")
            if not p["illustrative_only"]:
                keys = ("annotation_manifest_sha256",) if p["candidate_source"] == "gt" else ("producer_run_id","checkpoint_sha256","decoder_config_sha256")
                require(all(p["provenance"].get(k) for k in keys), "missing candidate provenance")

    def load(self, image_id):
        im, row = self.images[image_id], self.rows[image_id]
        path = safe_path(self.manifest_path.parent, im["file_name"])
        require(sha256(path) == im["sha256"], "image hash mismatch")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        require(image.size == (im["width"], im["height"]), "decoded image dimensions differ")
        result = []
        for c in row["candidates"]:
            path = safe_path(self.path.parent, c["mask_reference"])
            require(sha256(path) == c["mask_sha256"], "mask hash mismatch")
            mask = decode_mask(read_json(path), image.height, image.width)
            require(c["bbox_xyxy"] == mask_box(mask), "mask/bbox disagreement")
            result.append(Candidate(c["candidate_id"], c["category_id"], c["score"],
                                    tuple(c["bbox_xyxy"]), mask, c["decoder_query_id"]))
        return image, result

def crop_regions(image, candidates):
    # Internal boxes are exact integer mask bounds. No implicit clipping or expansion.
    result = []
    for c in candidates:
        x1,y1,x2,y2 = c.bbox_xyxy
        require(all(float(v).is_integer() for v in c.bbox_xyxy), "noninteger mask box")
        require(0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height, "invalid crop")
        result.append(image.crop(tuple(int(v) for v in c.bbox_xyxy)))
    return result
