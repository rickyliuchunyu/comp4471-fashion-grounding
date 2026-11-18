"""Versioned immutable normalized feature caches; no trained ranking outputs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np
from fashion_grounding.data.schema import read_json, sha256, require

CACHE_VERSION = "grounding-features-v1"

def cache_key(spec):
    payload = json.dumps({"version":CACHE_VERSION,"spec":spec},ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def check_features(values, rows, dimension):
    values = np.asarray(values)
    require(values.dtype == np.float32 and values.shape == (rows,dimension), "feature dtype/shape mismatch")
    require(np.isfinite(values).all(), "nonfinite feature")
    if rows:
        require(np.allclose(np.linalg.norm(values,axis=1),1,rtol=1e-4,atol=1e-5), "features not unit length")
    return values

class FeatureCache:
    def __init__(self, root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    def get(self, spec, rows, dimension):
        directory=self.root/cache_key(spec)
        if not directory.exists():
            return None
        meta=read_json(directory/"metadata.json")
        require(meta["version"] == CACHE_VERSION and meta["spec"] == spec, "cache identity mismatch")
        require(sha256(directory/"features.npy")==meta["sha256"], "cache bytes corrupted")
        return check_features(np.load(directory/"features.npy",allow_pickle=False),rows,dimension)

    def put(self, spec, values):
        values=check_features(values,*values.shape)
        target=self.root/cache_key(spec)
        if target.exists():
            old=self.get(spec,*values.shape)
            require(np.array_equal(old,values), "refusing to replace immutable cache")
            return
        temporary=Path(tempfile.mkdtemp(prefix=".pending-",dir=self.root))
        try:
            np.save(temporary/"features.npy",values,allow_pickle=False)
            meta={"version":CACHE_VERSION,"spec":spec,"sha256":sha256(temporary/"features.npy")}
            (temporary/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
            os.rename(temporary,target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

def visual_spec(pack, image_id, encoder):
    return {"kind":"image","candidate_manifest_sha256":pack.digest,
            "candidate_set_id":pack.package["candidate_set_id"],
            "candidate_source":pack.package["candidate_source"],
            "provenance":pack.package["provenance"],
            "image_sha256":pack.images[image_id]["sha256"],
            "candidate_ids":[c["candidate_id"] for c in pack.rows[image_id]["candidates"]],
            "encoder":encoder,"crop":"exact-mask-bbox-rgb-v1"}

def text_spec(text, language, encoder):
    return {"kind":"text","original_text":text,"language":language,
            "translation":"none","encoder":encoder,"length_policy":"reject-overlength"}
