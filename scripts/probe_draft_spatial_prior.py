"""Run the frozen M2 plus B2 position prior on unreviewed GT-candidate dev queries."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from fashion_grounding.data.schema import read_json, sha256
from fashion_grounding.encoders.siglip import SiglipEncoder
from fashion_grounding.grounding.geometry import geometry_features
from fashion_grounding.grounding.inputs import CandidatePack, crop_regions, inference_query
from fashion_grounding.grounding.predict import rank_scores
from fashion_grounding.grounding.rules import shared_parse, position_scores
from fashion_grounding.grounding.spatial import make_head, score_features


def run(config, pack_dir, output):
    cfg = read_json(config)
    root = Path(cfg["previous_experiment"])
    signature = read_json(Path(cfg["previous_inference"]) / "run_metadata.json")["encoder"]
    meta = read_json(root / "run_metadata.json")["models"]["M2-clean1000"]
    assert sha256(meta["checkpoint"]) == meta["checkpoint_sha256"]
    weight = read_json(root / "B2-dev-selection.json")["selected_weight"]
    lexicon_path = config.parent.parent / cfg["lexicon"]
    assert sha256(lexicon_path) == cfg["sha256"]["lexicon"]
    lexicon = read_json(lexicon_path)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    encoder = SiglipEncoder(cfg["snapshot"], signature["revision"], "cpu", 16)
    state = torch.load(meta["checkpoint"], map_location="cpu")
    assert encoder.signature == signature == state["encoder"]
    model = make_head("M2", state["dimension"])
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    pack = CandidatePack(pack_dir / "candidates_gt.json")
    assert pack.package["candidate_source"] == "gt"
    raw = [json.loads(s) for s in (pack_dir / "queries_draft.jsonl").read_text().splitlines()]
    assert all(q["split"] == "dev" and q["review_status"] == "needs_revision" for q in raw)
    queries = [inference_query(q) for q in raw]
    text_vectors = encoder.encode_texts([q.query for q in queries])
    selected = {}
    for iid in dict.fromkeys(q.image_id for q in queries):
        image, cs = pack.load(iid)
        crops = crop_regions(image, cs)
        visual = encoder.encode_images(crops)
        geom = geometry_features(cs, image.size)
        for crop in crops:
            crop.close()
        image.close()
        for i, q in enumerate(queries):
            if q.image_id != iid:
                continue
            text = text_vectors[i:i + 1]
            base = rank_scores(visual, text, signature["logit_scale"], signature["logit_bias"])
            learned = score_features(model, visual, text, geom, base)
            bonus = weight * position_scores(cs, shared_parse(q.query, q.language, lexicon))
            methods = {"B2": base + bonus, "M2": learned, "hybrid": learned + bonus}
            selected[q.query_id] = {k: cs[int(np.argmax(v))].candidate_id for k, v in methods.items()}
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for q in raw:
        targets = [c["candidate_id"] for c in pack.rows[q["image_id"]]["candidates"]
                   if c["source_annotation_id"] == q["target_instance_id"]]
        assert len(targets) == 1
        picks = selected[q["query_id"]]
        records.append({**{k: q[k] for k in ("query_id", "query_types", "image_id", "semantic_query_id")},
                        "selected": picks, "draft_target": targets[0],
                        "agreement": {k: v == targets[0] for k, v in picks.items()}})
    (output / "draft_diagnostic.json").write_text(json.dumps({
        "scope": "agreement with machine draft using GT candidates",
        "checkpoint_sha256": meta["checkpoint_sha256"], "weight": weight,
        "query_sha256": sha256(pack_dir / "queries_draft.jsonl"), "candidate_sha256": pack.digest,
        "rows": records}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "pack-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    run(args.config, args.pack_dir, args.output)
