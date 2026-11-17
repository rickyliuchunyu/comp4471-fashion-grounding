import unittest,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from publish_language_expansion import validate_decisions,asset_copy
class ExpansionGuards(unittest.TestCase):
 def setUp(self):
  self.s=[dict(index=1,image_id=2)]
  self.d={"1":dict(texts={"zh":"黑色包","en":"the black bag"},type="appearance",reason="visual inspection")}
 def test_complete_bilingual(self):validate_decisions(self.s,self.d)
 def test_missing_decision(self):
  with self.assertRaises(ValueError):validate_decisions(self.s,{})
 def test_missing_language(self):
  del self.d["1"]["texts"]["en"]
  with self.assertRaises(ValueError):validate_decisions(self.s,self.d)
 def test_relation_rejected(self):
  self.d["1"]["texts"]["en"]="the bag held in the hand"
  with self.assertRaises(ValueError):validate_decisions(self.s,self.d)
 def test_rejection_needs_reason(self):
  with self.assertRaises(ValueError):validate_decisions(self.s,{"1":{"reject":True}})
 def test_rejection_preserved(self):validate_decisions(self.s,{"1":{"reject":True,"reason":"one trouser leg"}})
 def test_asset_collision(self):
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/"a";b=Path(t)/"b";a.write_bytes(b"one");b.write_bytes(b"two")
   with self.assertRaises(ValueError):asset_copy(a,b)
   self.assertEqual(b.read_bytes(),b"two")
if __name__=="__main__":unittest.main()
