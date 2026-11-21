"""New descriptive statistics only; no data/model/old smoke suite."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('training_distribution',
    Path(__file__).resolve().parents[1] / 'scripts/analyze_segmentation_training_distribution.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistributionTests(unittest.TestCase):
    def test_area_boundaries_and_nested_tiny(self):
        areas = [64, 65, 256, 257, 1024, 1025, 9216, 9217]
        rows = [dict(image_id=1, area=a) for a in areas]
        result = module.summarize(rows, {1:dict(group_id='a', width=100, height=100)}, 8)
        self.assertEqual([result[k] for k in ('le64','gt64_le256','gt256_le1024',
                                            'gt1024_le9216','gt9216')], [1,2,2,2,1])
        self.assertEqual((result['small_le1024'], result['tiny_le256']), (5,3))
        self.assertEqual(result['relative_area_le1pct'], 2)
        self.assertEqual(result['area_p50'], 640.5)
        with self.assertRaises(ValueError): module.area_band(0)

    def test_image_group_deduplication_and_absent_fine_class(self):
        source = [dict(source_id=i, source_name=n, action='map', project_category_id=8)
                  for i,n in [(19,'belt'),(20,'leg warmer')]]
        images = [dict(id=i, group_id='same-group', width=100, height=100) for i in [1,2]]
        anns = [dict(image_id=i, area=100, source_category_id=19,
                     source_category_name='belt', category_id=8) for i in [1,1,2]]
        ignored = dict(anns[0], iscrowd=1)
        result = module.analyze(dict(images=images, annotations=anns+[ignored],
            categories=[dict(id=8,name='accessory')]), dict(fashionpedia=source))
        self.assertEqual((result['total']['instances'],result['total']['images'],
                          result['total']['groups']), (3,2,1))
        self.assertEqual(result['excluded'], {'crowd_or_ignore':1})
        empty = result['source_categories'][1]
        self.assertEqual(empty['instances'], 0)
        self.assertIsNone(empty['area_p50'])
        self.assertIsNone(empty['small_fraction'])
        anns[0]['category_id'] = 7
        with self.assertRaises(ValueError):
            module.analyze(dict(images=images,annotations=anns,categories=[]),dict(fashionpedia=source))


if __name__ == '__main__':
    unittest.main()
