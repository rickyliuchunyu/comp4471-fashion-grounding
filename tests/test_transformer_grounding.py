"""Regression checks for the native Transformer release and inference route."""
import copy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fashion_grounding.data.schema import sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.inputs import CandidatePack, inference_query
from fashion_grounding.grounding.predict import execute
from fashion_grounding.grounding.transformer import AttributeTransformer, load_transformer
from test_grounding import FakeEncoder, ROOT


class TransformerTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.encoder = FakeEncoder()
        signature = {"dimension": 2, **self.encoder.signature}
        self.encoder.signature = signature
        model = AttributeTransformer(2)
        with torch.no_grad():
            model.cue_map[-1, 0] = 1
            model.gains.zero_()
        paths = {
            "encoder": self.root / "encoder.json",
            "category_map": self.root / "category.npy",
            "checkpoint": self.root / "model.pt",
        }
        write_json(paths["encoder"], signature)
        np.save(paths["category_map"], np.ones((3, 8), dtype=np.float32))
        torch.save({"dimension": 2, "model": model.state_dict()}, paths["checkpoint"])
        assets = {k: {"path": str(path), "sha256": sha256(path)} for k, path in paths.items()}
        self.release = {
            "encoder": assets["encoder"], "category_map": assets["category_map"],
            "checkpoints": {"42": assets["checkpoint"]}, "default_seed": 42,
            "calibration": {"transformer-full": [5, 0, 10], "transformer-category-only": [5, 0, 0]},
        }

    def test_public_prediction_uses_geometry_once_and_category_only_disables_location(self):
        pack = CandidatePack(ROOT / "examples/candidates_gt.json", allow_illustrative=True)
        query = inference_query({
            "schema_version": 3, "query_id": "transformer", "image_id": next(iter(pack.rows)),
            "query": "the rightmost bag", "language": "en", "split": "dev", "illustrative_only": True,
        })
        query = replace(query, split=pack.images[query.image_id]["split"])
        cache = FeatureCache(self.root / "cache")
        ref = {"path": "inputs/candidates_gt.json", "sha256": pack.digest}
        choices = []
        for method in self.release["calibration"]:
            ranker, metadata = load_transformer(self.release, self.encoder.signature, method)
            candidate_ids = [r["candidate_id"] for r in pack.rows[query.image_id]["candidates"]]
            rows = execute(pack, [query], self.encoder, cache, "test", method, ref, transformer=ranker)
            self.assertIsNone(rows[0][1])
            self.assertEqual(rows[0][0]["method_id"], method)
            self.assertEqual(metadata["seed"], 42)
            choices.append(candidate_ids.index(rows[0][0]["selected_candidate"]["candidate_id"]))
        self.assertEqual(choices, [1, 0])
        self.assertEqual((self.encoder.image_calls, self.encoder.text_calls), (1, 1))

    def test_incompatible_encoder_and_changed_checkpoint_fail(self):
        with self.assertRaisesRegex(ValueError, "released encoder signature"):
            load_transformer(self.release, {"dimension": 3}, "transformer-full")
        changed = copy.deepcopy(self.release)
        changed["checkpoints"]["42"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint hash differs"):
            load_transformer(changed, self.encoder.signature, "transformer-full")

    def test_reference_cache_is_read_only(self):
        spec = {"kind": "test"}
        features = np.eye(2, dtype=np.float32)
        reference = FeatureCache(self.root / "source")
        reference.put(spec, features)
        cache = FeatureCache(self.root / "consumer", references=[reference.root])
        np.testing.assert_array_equal(cache.get(spec, 2, 2), features)
        self.assertEqual(list(cache.root.iterdir()), [])
        cache.put({"kind": "new"}, features)
        self.assertEqual(len(list(reference.root.iterdir())), 1)
