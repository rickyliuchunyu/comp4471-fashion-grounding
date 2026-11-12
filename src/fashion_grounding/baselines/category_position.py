"""B0: select by parsed category and unique box-center extreme only."""
from fashion_grounding.data.query_rules import parse_query

def select_target(query, language, candidates, lexicon, image_status="ok"):
    if image_status == "inference_error":
        return {"status":"inference_error","selected_candidate":None}
    if not candidates:
        return {"status":"no_candidates","selected_candidate":None}
    parsed = parse_query(query, language, lexicon)
    failed = {"status":"rule_unsupported","selected_candidate":None}
    if parsed.category_status != "unique" or parsed.direction_status == "conflict" or parsed.unsupported_operator or parsed.unconsumed_spans:
        return failed
    subset = [c for c in candidates if c["category_id"] == parsed.category_id]
    if not subset:
        return failed
    if parsed.direction:
        axis = 0 if parsed.direction in ("left","right") else 1
        centers = [(c["bbox_xyxy"][axis]+c["bbox_xyxy"][axis+2])/2 for c in subset]
        extreme = (min if parsed.direction in ("left","top") else max)(centers)
        subset = [c for c,center in zip(subset,centers) if center == extreme]
    if len(subset) != 1:
        return failed
    return {"status":"success","selected_candidate":{"candidate_id":subset[0]["candidate_id"],"grounding_score":None}}
