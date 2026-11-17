import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_reviewed_dev import bootstrap
class DevBootstrapTests(unittest.TestCase):
    def test_translations_stay_in_same_image_cluster(self):
        rows=[{"query_id":str(i),"image_id":1,"correct":v} for i,v in enumerate([True,False])]
        result=bootstrap(rows)
        self.assertEqual(result["ci95"],[.5,.5])
        self.assertEqual(result["images"],1)
    def test_paired_difference_joins_query_id_not_order(self):
        a=[{"query_id":"a","image_id":1,"correct":True},{"query_id":"b","image_id":2,"correct":False}]
        b=list(reversed(a))
        self.assertEqual(bootstrap(a,b)["ci95"],[0.,0.])
    def test_estimate_is_query_weighted_not_mean_image_accuracy(self):
        rows=[{"query_id":str(i),"image_id":1 if i<3 else 2,"correct":i<3} for i in range(4)]
        self.assertEqual(bootstrap(rows)["estimate"],.75)
if __name__=="__main__":unittest.main()
