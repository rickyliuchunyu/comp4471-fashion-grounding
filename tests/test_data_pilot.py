import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.convert import convert_image, draft_queries
from fashion_grounding.data.schema import decode_mask, encode_mask, mask_box, safe_path, validate_queries


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "image.png"
        Image.new("RGB", (5, 4)).save(self.path)
        self.image = {"id": 0, "width": 5, "height": 4}
        self.mapping = {0: {"action": "map", "project_category_id": 1, "source_name": "top"}, 11: {"action": "exclude_image"}, 32: {"action": "exclude_annotation"}}
        self.mask = np.zeros((4, 5), dtype=bool)
        self.mask[1:3, 1:4] = True
        self.ann = {"id": 0, "category_id": 0, "segmentation": encode_mask(self.mask), "iscrowd": 0}

    def test_hand_computed_geometry_and_zero_id(self):
        anns, masks = convert_image(self.image, [self.ann], self.mapping, self.path)
        self.assertEqual(anns[0]["id"], 1)
        self.assertEqual(anns[0]["image_id"], 1)
        self.assertEqual(anns[0]["source_annotation_id"], 0)
        self.assertEqual(anns[0]["bbox"], [1, 1, 3, 2])
        self.assertEqual(anns[0]["area"], 6)
        self.assertEqual(mask_box(masks[1]), [1, 1, 4, 3])

    def test_part_annotation_only_is_removed(self):
        part = {**self.ann, "id": 1, "category_id": 32}
        anns, _ = convert_image(self.image, [self.ann, part], self.mapping, self.path)
        self.assertEqual(len(anns), 1)

    def test_jumpsuit_excludes_whole_image(self):
        with self.assertRaisesRegex(ValueError, "jumpsuit"):
            convert_image(self.image, [self.ann, {**self.ann, "category_id": 11}], self.mapping, self.path)

    def test_crowd_preserved_but_not_a_query_target(self):
        anns, _ = convert_image(self.image, [{**self.ann, "iscrowd": 1}], self.mapping, self.path)
        self.assertEqual(anns[0]["iscrowd"], 1)
        self.assertEqual(draft_queries({"image_id": 1, "width": 5}, anns), [])

    def test_disconnected_instance_stays_one_instance(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[0, 0] = mask[3, 4] = True
        anns, _ = convert_image(self.image, [{**self.ann, "segmentation": encode_mask(mask)}], self.mapping, self.path)
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0]["area"], 2)
        self.assertEqual(anns[0]["bbox"], [0, 0, 5, 4])

    def test_column_major_uncompressed_rle(self):
        expected = np.array([[0, 1, 0], [1, 1, 0]], dtype=bool)
        self.assertTrue(np.array_equal(decode_mask({"size": [2, 3], "counts": [1, 3, 2]}, 2, 3), expected))

    def test_bad_rle_empty_mask_and_outside_polygon_rejected(self):
        for seg in ({"size": [2, 3], "counts": [1, 2]}, {"size": [2, 3], "counts": [6]}, [[-1, 0, 2, 0, 2, 1]]):
            with self.assertRaises(ValueError):
                decode_mask(seg, 2, 3)

    def test_image_dimensions_not_silently_corrected(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            convert_image({**self.image, "width": 9}, [self.ann], self.mapping, self.path)

    def test_paths_cannot_escape(self):
        for value in ("../secret", "/absolute", "C:/secret", "https://host/file", "sub/../../secret", "sub\\file"):
            with self.assertRaises(ValueError):
                safe_path(self.temp.name, value)

    def test_drafts_rejected_by_formal_evaluation_gate(self):
        anns, _ = convert_image(self.image, [self.ann], self.mapping, self.path)
        im = {"image_id": 1, "width": 5, "split": "train"}
        qs = draft_queries(im, anns)
        validate_queries(qs, {1: im}, {1: anns[0]})
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            validate_queries(qs, {1: im}, {1: anns[0]}, accepted_only=True)

    def test_query_wrong_image_split_duplicate_and_self_review(self):
        anns, _ = convert_image(self.image, [self.ann], self.mapping, self.path)
        im = {"image_id": 1, "width": 5, "split": "train"}
        q = draft_queries(im, anns)[0]
        cases = [[{**q, "split": "test"}], [q, q], [{**q, "review_status": "accepted", "reviewer": q["annotator"]}]]
        for rows in cases:
            with self.assertRaises(ValueError):
                validate_queries(rows, {1: im}, {1: anns[0]})
        with self.assertRaisesRegex(ValueError, "invalid query target"):
            validate_queries([q], {1: im}, {1: {**anns[0], "image_id": 2}})


if __name__ == "__main__":
    unittest.main()
