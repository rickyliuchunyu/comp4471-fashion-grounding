"""Only the newly added geometry/association join; no model or old smoke."""
import copy
import tempfile
import unittest
from pathlib import Path

from fashion_grounding.data.schema import write_json, sha256
from fashion_grounding.segmentation.pipeline import resolve_outputs, evaluate_saved_associations


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        top = {'size': [2, 2], 'counts': [0, 1, 1, 1, 1]}
        bottom = {'size': [2, 2], 'counts': [1, 1, 1, 1]}
        full = {'size': [2, 2], 'counts': [0, 4]}
        candidates = []
        for i, mask in enumerate((top, bottom, full)):
            path = self.root/f'{i}.json'
            write_json(path, mask)
            candidates.append(dict(candidate_id=str(i), category_id=1, bbox_xyxy=[0, 0, 2, 2],
                mask_reference=path.name, mask_sha256=sha256(path), decoder_query_id=i))
        self.package = dict(candidate_set_id='fixed', candidate_source='predicted', images=[
            dict(image_id=1, image_size_wh=[2, 2], status='ok', candidates=candidates[:2]),
            dict(image_id=2, image_size_wh=[2, 2], status='ok', candidates=candidates[2:])])
        self.ref = dict(path='inputs/candidates.json', sha256='fixed_hash')
        self.annotations = [dict(id=10, image_id=1, category_id=1, iscrowd=0, segmentation=top),
            dict(id=11, image_id=1, category_id=1, iscrowd=0, segmentation=bottom),
            dict(id=20, image_id=2, category_id=2, iscrowd=0, segmentation=full)]
        self.queries, self.predictions, rows = [], [], []
        for i, (iid, aid, matched, selected) in enumerate([(1,10,'0','0'),(1,11,'1','0'),(2,20,None,'2'),(1,10,'0',None)]):
            q = dict(query_id=str(i), image_id=iid, target_instance_id=aid, semantic_query_id=str(i),
                     query='original text', language='en', split='dev', review_status='needs_revision')
            self.queries.append(q)
            self.predictions.append(dict(schema_version=3, query_id=q['query_id'], image_id=iid,
                query=q['query'], language='en', split='dev', run_id='r', method_id='B1',
                candidate_source='predicted', candidate_set_id='fixed', candidate_manifest=self.ref,
                status='success' if selected is not None else 'inference_error',
                selected_candidate={'candidate_id': selected, 'grounding_score': 1.} if selected else None))
            rows.append(dict(query_id=str(i), image_id=iid, target_instance_id=aid,
                matched_candidate_id=matched, covered=matched is not None, geometric_recalled=True))
        self.coverage = dict(rows=rows, summary={'matched_candidate_coverage': .75})

    def evaluate(self):
        return evaluate_saved_associations(self.queries, self.predictions, self.coverage,
            self.package, self.root/'candidates.json', self.annotations)

    def test_original_geometry_and_wrong_image_rejection(self):
        before = copy.deepcopy(self.package)
        out = resolve_outputs(self.predictions, self.package, self.ref)
        self.assertEqual(out[0]['geometry']['bbox_xyxy'], before['images'][0]['candidates'][0]['bbox_xyxy'])
        self.assertEqual(out[0]['geometry']['mask_reference'], 'inputs/0.json')
        self.assertIsNone(out[3]['geometry'])
        self.assertEqual(self.package, before)
        self.predictions[0]['selected_candidate']['candidate_id'] = '2'
        with self.assertRaises(ValueError): resolve_outputs(self.predictions, self.package, self.ref)

    def test_manual_metrics_and_review_labels_unchanged(self):
        before = copy.deepcopy(self.queries)
        s = self.evaluate()['summary']
        self.assertEqual(s['predicted_selection_accuracy'], .25)
        self.assertEqual(s['conditional_selection_accuracy'], 1/3)
        self.assertEqual(s['mean_target_mask_iou'], .5)
        self.assertEqual(s['success_at_0_5'], .5)  # Wrong class geometry succeeds, association does not.
        self.assertEqual(s['queries'], 4)  # Failure stays in denominator.
        self.assertEqual(self.queries, before)

    def test_incomplete_and_changed_geometry_fail_closed(self):
        old = self.predictions.pop()
        with self.assertRaises(ValueError): self.evaluate()
        self.predictions.append(old)
        (self.root/'0.json').write_text('{}')
        with self.assertRaises(ValueError): self.evaluate()


if __name__ == '__main__': unittest.main()
