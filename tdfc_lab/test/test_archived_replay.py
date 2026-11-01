from pathlib import Path
import unittest
from tdfc_lab.adapter import load_task
from tdfc_lab.offline import replay


class ArchivedReplayTests(unittest.TestCase):
    def test_three_frozen_measured_observation_trajectories(self):
        for shape in ('ring', 'tilted_ring', 'helix'):
            with self.subTest(shape=shape):
                result = replay(Path(__file__).parent/'fixtures'/f'{shape}.npz', load_task(shape))
                self.assertTrue(result['expected_compared'])
                self.assertEqual(result['steps'], 200)
                self.assertEqual(result['maximum_command_error'], 0.)


if __name__ == '__main__':
    unittest.main()
