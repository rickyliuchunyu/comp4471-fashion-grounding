"""Only new offline derivation: matching tie/competition and cause precedence."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_50k_residual_errors import matrix_associate, residual_cause
from fashion_grounding.evaluation.matching import associate


class ResidualDerivation(unittest.TestCase):
    def test_ties_and_class_match_existing_mask_matcher(self):
        a=np.array([[1,1],[0,0]],dtype=bool)
        b=np.array([[0,0],[1,1]],dtype=bool)
        anns=[dict(id=20,category_id=6,iscrowd=0),dict(id=10,category_id=6,iscrowd=0),dict(id=30,category_id=7,iscrowd=0)]
        cs=[dict(candidate_id='b',score=.8,category_id=6),dict(candidate_id='a',score=.8,category_id=6),dict(candidate_id='c',score=.9,category_id=8),dict(candidate_id='d',score=.7,category_id=7)]
        cm={'a':a,'b':a,'c':b,'d':b};gm={20:a,10:a,30:b}
        overlaps=np.array([[1,1,0],[1,1,0],[0,0,1],[0,0,1]],dtype=float)
        result=matrix_associate(cs,anns,overlaps)
        self.assertEqual(result,{'a':10,'b':20,'d':30})
        self.assertEqual(result,associate(cs,anns,cm,gm))

    def test_threshold_and_empty_and_competition(self):
        cs=[dict(candidate_id='a',score=.7,category_id=6)]
        anns=[dict(id=1,category_id=6,iscrowd=0),dict(id=2,category_id=6,iscrowd=0)]
        self.assertEqual(matrix_associate(cs,anns,np.array([[.5,.49]])),{'a':1})
        self.assertEqual(matrix_associate(cs,anns,np.array([[.5,.49]]),.75),{})
        self.assertEqual(matrix_associate([],anns,np.zeros((0,2))),{})

    def test_residual_partition_precedence(self):
        v=lambda a,b:dict(best_same=dict(iou=a),best_other=dict(iou=b))
        self.assertEqual(residual_cause(v(.7,.8),v(.7,.8),True),'matched50')
        self.assertEqual(residual_cause(v(.7,.8),v(.7,.8),False),'one_to_one_contention')
        self.assertEqual(residual_cause(v(.4,.8),v(.7,.8),False),'score_filtered')
        self.assertEqual(residual_cause(v(.4,.4),v(.4,.8),False),'wrong_class_only')
        self.assertEqual(residual_cause(v(0,0),v(.1,.05),False),'partial_or_mixed_geometry')
        self.assertEqual(residual_cause(v(0,0),v(.09,.05),False),'no_saved_mask_iou_ge_010')


if __name__=='__main__':unittest.main()
