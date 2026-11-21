"""Boundary cases specific to the new full-population / small-target sampler."""
import unittest

from fashion_grounding.segmentation.full_sampling import build_order


def image(i, group):
    return dict(image_id=i, group_id=group, sha256=str(i), source_image_id=str(i - 1),
                split='train', source_split='official_train2020')


def annotation(i, size=1024, crowd=0):
    return dict(image_id=i, area=size, source_category_name='sock', iscrowd=crowd)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.images = [image(1, 'shared'), image(2, 'shared'), image(3, 'large'), image(4, 'crowd')]
        self.anns = [annotation(1), annotation(2), annotation(3, 1025), annotation(4, crowd=1)]
        self.protection = dict(images=[], forbidden_splits=['train', 'test'], distinct_images=0)
        self.config = dict(seed=42, training_images=4, sampling=dict(
            source_names=['sock'], target_draws=2, max_area=1024, max_extra_per_group=2))

    def build(self):
        return build_order(self.images, self.anns, self.protection, self.config)

    def test_inclusive_area_crowd_dedup_and_deterministic_order(self):
        order, report = self.build()
        self.assertEqual(set(order), {1, 2, 3, 4})
        self.assertEqual(sorted([order[2], order[5]]), [1, 2])
        self.assertEqual(report['max_extra_per_group'], 2)
        self.images.reverse()
        self.anns.append(annotation(1))
        self.assertEqual(self.build(), (order, report))

    def test_shared_group_cannot_be_oversampled_across_members(self):
        self.config['sampling']['target_draws'] = 3
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            self.build()

    def test_protected_group_blocks_even_a_different_image(self):
        self.protection.update(distinct_images=1, images=[dict(image_id=55, sha256='x',
                               group_id='shared', source_key='val-2')])
        with self.assertRaisesRegex(ValueError, 'Protected'):
            self.build()
