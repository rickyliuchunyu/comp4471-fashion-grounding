"""Shared text-only parser for B0 and B2; no target labels enter this API."""
import re
import unicodedata
from dataclasses import dataclass

@dataclass(frozen=True)
class ParsedQuery:
    lexicon_version: str
    normalized_text: str
    category_id: int | None
    direction: str | None
    category_status: str
    direction_status: str
    unsupported_operator: bool
    unconsumed_spans: list[str]

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def parse_query(query, language, lexicon):
    if language not in ("zh", "en") or not isinstance(query, str):
        raise ValueError("expected zh/en and text")
    text = normalize(query)
    def matches(term):
        term = normalize(term)
        pattern = re.escape(term)
        if language == "en":
            pattern = r"(?<!\w)" + pattern + r"(?!\w)"
        return list(re.finditer(pattern, text))
    unsupported = any(matches(t) for t in lexicon["unsupported_operator_terms"][language])
    terms = []
    for row in lexicon["category_aliases"]:
        terms.extend((t, "category", row["category_id"]) for t in row[language])
    for direction, aliases in lexicon["directions"].items():
        terms.extend((t, "direction", direction) for t in aliases[language])
    terms.extend((t, "filler", None) for t in lexicon["filler_tokens"][language])
    spans = [(m.start(), m.end(), kind, value) for term, kind, value in terms for m in matches(term)]
    used = [False] * len(text)
    found = {"category": set(), "direction": set()}
    for start, end, kind, value in sorted(spans, key=lambda s: (-(s[1]-s[0]), s[0], s[2])):
        if any(used[start:end]):
            continue
        used[start:end] = [True] * (end-start)
        if kind in found:
            found[kind].add(value)
    remainder = "".join(" " if used[i] or unicodedata.category(c).startswith("P") else c for i,c in enumerate(text))
    def status(values):
        return "missing" if not values else "unique" if len(values)==1 else "conflict"
    return ParsedQuery(lexicon["lexicon_version"], text,
        next(iter(found["category"])) if len(found["category"])==1 else None,
        next(iter(found["direction"])) if len(found["direction"])==1 else None,
        status(found["category"]), status(found["direction"]), unsupported, remainder.split())
