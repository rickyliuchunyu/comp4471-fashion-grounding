"""Audit exported human decisions without silently accepting or rewriting targets."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
from .schema import read_json, require, sha256, unique_index, validate_package, write_json

def audit_review(manifest, annotations, queries, review, disclosed_images=()):
    require(review["manifest_id"] == manifest["manifest_id"], "review manifest mismatch")
    require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()), "reviewer required")
    images = unique_index(manifest["images"], "image_id")
    disclosed_images = set(disclosed_images) | set(review.get("disclosed_image_ids", []))
    require(disclosed_images <= set(images), "unknown disclosed image")
    require(review.get("review_format_version", 1) in (1, 2), "unsupported review format")
    anns = unique_index(annotations, "id")
    unique_index(queries, "query_id")
    groups = defaultdict(list)
    for q in queries:
        groups[q["semantic_query_id"]].append(q)
    decisions = unique_index(review["rows"], "semantic_query_id")
    require(set(decisions) <= set(groups), "unknown semantic query")
    rows = []
    for sid, qs in sorted(groups.items()):
        q = qs[0]
        require(len(qs) == 2 and {x["language"] for x in qs} == {"zh", "en"}, "expected one bilingual pair")
        require(len({(x["image_id"], x["target_instance_id"], x["split"]) for x in qs}) == 1, "translation target mismatch")
        require(q["image_id"] in images and q["split"] == images[q["image_id"]]["split"], "query split mismatch")
        target = anns[q["target_instance_id"]]
        require(target["image_id"] == q["image_id"] and not target["iscrowd"], "invalid draft target")
        decision = decisions.get(sid)
        selected = None
        note = ""
        edits = None
        checks = None
        if decision is None:
            status = "not_reviewed"
        else:
            choice = decision["decision"]
            require(choice in {"human_selected", "needs_revision", "not_reviewed"}, "invalid decision")
            selected = decision["selected_annotation_id"]
            note = decision.get("note", "")
            require(isinstance(note, str), "note must be text")
            edits = decision.get("edited_queries")
            checks = decision.get("checks")
            if review.get("review_format_version") == 2:
                require(isinstance(edits, dict) and set(edits) == {"zh", "en"} and all(isinstance(v, str) for v in edits.values()), "invalid edited bilingual text")
                require(isinstance(checks, dict) and set(checks) == {"unique", "translation", "mask"} and all(type(v) is bool for v in checks.values()), "invalid review checks")
            if choice == "human_selected":
                require(type(selected) is int and selected in anns, "invalid selected annotation")
                require(anns[selected]["image_id"] == q["image_id"] and not anns[selected]["iscrowd"], "selected target belongs to another image or crowd")
                if selected != q["target_instance_id"]:
                    status = "target_disagreement"
                elif q["image_id"] in disclosed_images:
                    status = "answer_disclosed"
                elif note.strip():
                    status = "agreement_with_note"
                else:
                    status = "target_agreement_only"
                if review.get("review_format_version") == 2:
                    if edits != {x["language"]:x["query"] for x in qs}:
                        status = "edited_query_pending_recheck"
                    elif not all(checks.values()):
                        status = "incomplete_quality_checks"
            else:
                require(selected is None, "non-selection must not carry target")
                status = choice
        same = sum(a["image_id"] == q["image_id"] and a["category_id"] == target["category_id"] and not a["iscrowd"] for a in annotations)
        rows.append({"semantic_query_id": sid, "query_ids": [x["query_id"] for x in qs], "image_id": q["image_id"], "queries": {x["language"]:x["query"] for x in qs}, "draft_target": q["target_instance_id"], "human_target": selected, "status": status, "note": note, "edited_queries": edits, "checks": checks, "answer_disclosed": q["image_id"] in disclosed_images, "target_agrees": selected == q["target_instance_id"] if selected is not None else None, "proposed_annotation_id": decision.get("proposed_annotation_id") if decision else None, "query_types": q["query_types"], "target_category_id": target["category_id"], "same_category_instance_count": same})
    counts = Counter(r["status"] for r in rows)
    types = Counter("+".join(sorted(r["query_types"])) for r in rows)
    return {"manifest_id": manifest["manifest_id"], "review_record_count": len(decisions), "semantic_queries": len(rows), "status_counts": dict(counts), "query_type_counts": dict(types), "single_category_target_count": sum(r["same_category_instance_count"] == 1 for r in rows), "appearance_or_combined_count": sum(bool(set(r["query_types"]) & {"appearance","combined"}) for r in rows), "accepted_promoted": 0, "limitations": ["Target agreement is not independent acceptance or proof of research quality.", "Same-category count does not guarantee visual ambiguity.", "Input package and original decisions are unchanged."], "rows": rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disclosed-image-id", type=int, action="append", default=[])
    args = parser.parse_args()
    require(not args.output.exists(), "audit output exists; choose a new run path")
    validate_package(args.package)
    qs = [json.loads(x) for x in (args.package/"queries_draft.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    review_input = read_json(args.review)
    if review_input.get("review_format_version") == 2:
        require(review_input.get("query_sha256") == sha256(args.package/"queries_draft.jsonl"), "review query hash mismatch")
    report = audit_review(read_json(args.package/"image_manifest.json"), read_json(args.package/"annotations_coco.json")["annotations"], qs, review_input, set(args.disclosed_image_id))
    report["provenance"] = {"review_sha256": sha256(args.review), "query_sha256": sha256(args.package/"queries_draft.jsonl"), "candidate_sha256": sha256(args.package/"candidates_gt.json"), "disclosed_image_ids": sorted(set(args.disclosed_image_id) | set(review_input.get("disclosed_image_ids", [])))}
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
