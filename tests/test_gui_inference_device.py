"""Exercise upload device routing without loading weights or running inference."""
import fcntl
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from segmentation_gui.predict_upload import GpuUnavailable, select_device


class UploadDeviceTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.coord = Path(tmp.name)

    def test_auto_holds_team_lock_only_while_using_cuda(self):
        with patch('with_gpu_lock.subprocess.check_output', side_effect=['', '0']), \
                patch('torch.cuda.is_available', return_value=True), ExitStack() as lease:
            self.assertEqual(select_device('auto', self.coord, lease), ('cuda', ''))
            with (self.coord / 'gpu.lock').open('a') as second:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (self.coord / 'gpu.lock').open('a') as second:
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_lock_contention_falls_back_only_in_auto(self):
        with (self.coord / 'gpu.lock').open('a') as other, ExitStack() as lease:
            fcntl.flock(other, fcntl.LOCK_EX)
            with patch('with_gpu_lock.subprocess.check_output') as probe:
                self.assertEqual(select_device('auto', self.coord, lease), ('cpu', '共享 GPU 正被占用'))
                with self.assertRaises(GpuUnavailable):
                    select_device('cuda', self.coord, lease)
                self.assertEqual(select_device('cpu', self.coord, lease), ('cpu', ''))
                probe.assert_not_called()

    def test_external_activity_or_unusable_cuda_releases_lock_before_cpu(self):
        for readings, available in ((['123', '0'], True), (['', '800'], True),
                                    (['', ''], True), (['', '0'], False)):
            with self.subTest(readings=readings, available=available), ExitStack() as lease, \
                    patch('with_gpu_lock.subprocess.check_output', side_effect=readings), \
                    patch('torch.cuda.is_available', return_value=available):
                device, reason = select_device('auto', self.coord, lease)
                self.assertEqual(device, 'cpu')
                self.assertTrue(reason)
                with (self.coord / 'gpu.lock').open('a') as second:
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main()
