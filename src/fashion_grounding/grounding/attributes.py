"""Learned text gates for category-relative position and candidate mask color."""
import numpy as np
import torch
from torch import nn

DIRECTIONS = ("none", "left", "right", "top", "bottom")


class AttributeHeads(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.direction = nn.Linear(dimension, len(DIRECTIONS))
        self.category = nn.Linear(dimension, 8)
        self.color_gate = nn.Linear(dimension, 1)
        self.color_weights = nn.Linear(dimension, 12)

    def evidence(self, text):
        """Prediction uses only the frozen text feature."""
        return (
            self.direction(text).softmax(-1),
            self.category(text).softmax(-1),
            self.color_gate(text).sigmoid().squeeze(-1),
            self.color_weights(text).tanh(),
        )


def ranking_components(head, examples):
    """Direction and color gates use predictions, with no parser or answer inputs."""
    with torch.inference_mode():
        batch = torch.from_numpy(np.concatenate([e["text"] for e in examples]))
        orientations, categories, color_gate, color_weights = (x.numpy() for x in head.evidence(batch))
    output = []
    for i, example in enumerate(examples):
        if not example["candidates"]:
            output.append(None)
            continue
        geometry = example["geometry"]
        if geometry.shape[1] != 24:
            raise ValueError("expected geometry and mask-only color features")
        candidate_cat = np.asarray([c.category_id - 1 for c in example["candidates"]])
        probabilities = orientations[i]
        decision = int(probabilities.argmax())
        is_direction = decision != 0 and probabilities[decision] >= .8
        category_prob = categories[i, candidate_cat]
        # Category-relative horizontal/vertical ranks are already stored in geometry.
        if is_direction:
            expected_direction = np.asarray([probabilities[2]-probabilities[1], probabilities[4]-probabilities[3]])
            position_bonus = ((2 * geometry[:, 10:12] - 1) @ expected_direction) * category_prob
            class_bonus = (category_prob - np.float32(.125)) * (1 - probabilities[0])
        else:
            position_bonus = np.zeros_like(geometry[:, 0])
            class_bonus = np.zeros_like(position_bonus)
        color_bonus = np.zeros_like(position_bonus)
        if not is_direction and color_gate[i] >= .75:
            color_bonus = (geometry[:, 12:] @ color_weights[i]) * color_gate[i] * category_prob
        output.append((position_bonus, class_bonus, color_bonus))
    return output


def add_components(base, components, weights):
    return [None if scores is None else scores + sum(w * s for w, s in zip(weights, terms))
            for scores, terms in zip(base, components)]



def hybrid_scores(head, examples, base, weights, lexicon, rule_weight):
    """Use a uniquely parsed direction when available, otherwise the learned scores."""
    from .rules import position_scores, shared_parse

    fallback = add_components(base, ranking_components(head, examples), weights)
    scores = []
    counts = {"rule": 0, "learned_or_unchanged": 0, "no_candidates": 0}
    for ex, logits, learned in zip(examples, base, fallback):
        if logits is None:
            scores.append(None)
            counts["no_candidates"] += 1
            continue
        query = ex["query"]
        parsed = shared_parse(query.query, query.language, lexicon)
        if parsed.direction_status == parsed.category_status == "unique" and not parsed.unsupported_operator:
            scores.append(logits + rule_weight * position_scores(ex["candidates"], parsed))
            counts["rule"] += 1
        else:
            scores.append(learned)
            counts["learned_or_unchanged"] += 1
    return scores, counts
