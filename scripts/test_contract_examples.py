"""Regression checks for artifact handoff failures; uses no model or real data."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ContractExamplesTest(unittest.TestCase):
    def run_checker(self, root):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_contract_examples.py"), "--root", str(root)],
            capture_output=True, text=True, check=False,
        )

    def test_valid_fixture(self):
        result = self.run_checker(ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_malformed_fixtures(self):
        cases = [
            ("missing query ID", "prediction.json", lambda o: o.pop("query_id"), "query_id mismatch"),
            ("unknown selected candidate", "prediction.json",
             lambda o: o["selected_candidate"].update(candidate_id="missing"), "Selected candidate missing"),
            ("mask bbox mismatch", "candidates_gt.json",
             lambda o: o["images"][0]["candidates"][0].update(bbox_xyxy=[0, 0, 1, 1]), "Mask/bbox mismatch"),
            ("invalid file hash", "image_manifest.json",
             lambda o: o["images"][0].update(sha256="0" * 64), "Hash mismatch"),
            ("changed candidate order", "candidates_predicted.json",
             lambda o: o["images"][0]["candidates"].reverse(), "Candidate order mismatch"),
            ("GT ID leaked into prediction", "candidates_predicted.json",
             lambda o: o["images"][0]["candidates"][0].update(source_annotation_id=700001), "leaks GT ID"),
            ("failure with selection", "prediction_no_candidates.json",
             lambda o: o.update(selected_candidate={"candidate_id": "x"}), "Failure must not select"),
            ("invalid RLE length", "fixtures/mask_1.json",
             lambda o: o.update(counts=[1]), "Invalid uncompressed RLE counts"),
        ]
        for title, filename, mutate, expected in cases:
            with self.subTest(title=title), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for folder in ("examples", "configs", "experiments"):
                    shutil.copytree(ROOT / folder, root / folder)
                path = root / "examples" / filename
                obj = json.loads(path.read_text())
                mutate(obj)
                path.write_text(json.dumps(obj))
                if filename == "fixtures/mask_1.json":
                    # Refresh the parent hash to reach RLE validation, not only integrity checking.
                    parent = root / "examples/candidates_gt.json"
                    package = json.loads(parent.read_text())
                    package["images"][0]["candidates"][0]["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    parent.write_text(json.dumps(package))
                result = self.run_checker(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
