"""B2 scoring only. Query parsing and the lexicon remain owned by data_eval."""
import numpy as np
from fashion_grounding.data.schema import require

def category_ranks(candidates, axis):
    require(axis in (0,1), "invalid rank axis")
    ranks = np.full(len(candidates), 0.5, dtype=np.float32)
    for category in {c.category_id for c in candidates}:
        indices = [i for i,c in enumerate(candidates) if c.category_id == category]
        if len(indices) <= 1:
            continue
        values = np.asarray([(candidates[i].bbox_xyxy[axis]+candidates[i].bbox_xyxy[axis+2])/2 for i in indices])
        for local,i in enumerate(indices):
            less = np.count_nonzero(values < values[local])
            equal = np.count_nonzero(values == values[local])
            ranks[i] = (less+(equal-1)/2)/(len(indices)-1)
    return ranks

def position_scores(candidates, parsed):
    """Accept the shared parser's mapping or ParsedQuery, never parse text here."""
    get = parsed.get if isinstance(parsed, dict) else lambda k: getattr(parsed,k)
    result = np.zeros(len(candidates), dtype=np.float32)
    if get("category_status") != "unique" or get("direction_status") != "unique" or get("unsupported_operator"):
        return result
    direction, category = get("direction"), get("category_id")
    require(direction in ("left","right","top","bottom") and category in range(1,9), "invalid parsed result")
    ranks = category_ranks(candidates, 0 if direction in ("left","right") else 1)
    sign = 1 if direction in ("right","bottom") else -1
    for i,c in enumerate(candidates):
        if c.category_id == category:
            result[i] = sign*(2*ranks[i]-1)
    return result

def shared_parse(query, language, lexicon):
    try:
        from fashion_grounding.data.query_rules import parse_query
    except ImportError as exc:
        raise RuntimeError("B2 needs wkq's shared data.query_rules.parse_query; no private parser fallback") from exc
    return parse_query(query, language, lexicon)
