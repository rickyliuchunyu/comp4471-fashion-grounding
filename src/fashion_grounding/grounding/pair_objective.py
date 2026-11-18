"""Pair supervision on same-image bilingual opposite descriptions."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from fashion_grounding.data.schema import require


@dataclass(frozen=True)
class PairMargin:
    """Encourage each query's own target to beat the other query's target."""

    # Map each supervised row to its counterpart's local target index.
    counter_targets: dict[int, int]
    score_source: str
    margin: float
    weight: float

    def __call__(self, model, batch, scores, indices):
        paired = [(row, self.counter_targets[index]) for row, index in enumerate(indices)
                  if index in self.counter_targets]
        require(bool(paired), "batch missing paired queries")
        if self.score_source == "geometry":
            scores = model.spatial_scores(batch[1], batch[5])
        rows, alternatives = zip(*paired)
        row_ids = torch.tensor(rows, device=scores.device)
        alternatives = torch.tensor(alternatives, device=scores.device)
        correct = batch[4][row_ids]
        gap = scores[row_ids, correct] - scores[row_ids, alternatives]
        return self.weight * F.softplus(self.margin - gap).mean()


def counter_targets(raw, samples, increment_ids):
    """Only use the published contrast groups. No text parser or guess at negative labels."""
    groups = {}
    for i, row in enumerate(raw):
        if row["query_id"] in increment_ids:
            key = (row["contrast_set_id"], row["language"])
            groups.setdefault(key, []).append(i)
    require(len(groups) == 172, "expected two languages for each of the 86 published contrast groups")
    targets = {}
    for group in groups.values():
        require(len(group) == 2, "need two targets of the same language")
        left, right = group
        require(raw[left]["image_id"] == raw[right]["image_id"], "different images")
        require(raw[left]["target_category_id"] == raw[right]["target_category_id"], "different categories")
        require(samples[left]["candidates"] == samples[right]["candidates"], "candidate order differs")
        a, b = samples[left]["target"], samples[right]["target"]
        require(a != b and len(samples[left]["candidates"]) >= 2, "need distinct competing targets")
        targets[left], targets[right] = b, a
    return targets
