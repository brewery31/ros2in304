import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from tdfc_lab.adapter import StateBuffer, CandidateController, load_task


class AdapterTests(unittest.TestCase):
    names = ('cf5', 'cf6', 'cfA', 'cfB')

    def test_name_mapping_preserves_hex_yaml_names(self):
        buffer = StateBuffer(self.names)
        self.assertEqual(tuple(buffer.history), self.names)
        with self.assertRaises(ValueError):
            buffer.add('cf10', 1., [0,0,1], [0,0,0])

    def test_asynchronous_interpolation_and_separate_ego_times(self):
        b = StateBuffer(self.names)
        for i, name in enumerate(self.names):
            for t in (1., 1.02+i*.001):
                b.add(name, t, [t, i, 1.2], [1,0,0])
        f = b.aligned(1.025)
        np.testing.assert_allclose(f['position'][:,0], 1.02)
        self.assertEqual(f['measurement_time'], 1.02)
        np.testing.assert_allclose(f['own_times'], [1.02,1.021,1.022,1.023])

    def test_future_stale_missing_and_nonfinite_states(self):
        b = StateBuffer(self.names)
        with self.assertRaises(ValueError): b.aligned(1.)
        for name in self.names: b.add(name, 1., [0,0,1], [0,0,0])
        for t in (.9, 1.2, float('nan')):
            with self.assertRaises(ValueError): b.aligned(t)
        with self.assertRaises(ValueError): b.add(self.names[0], 1., [0,0,1], [0,0,0])
        with self.assertRaises(ValueError): b.add(self.names[0], 1.02, [0,0,np.nan], [0,0,0])

    def test_fault_latches_instead_of_emitting_zero(self):
        c = CandidateController(load_task('ring'), self.names)
        frame = dict(position=np.zeros((4,3)), velocity=np.zeros((4,3)), measurement_time=0.,
                     own_velocities=np.zeros((4,3)), own_times=[0.]*4)
        with patch.object(c.controllers[0], 'step', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError): c.step(frame, 0.)
        with self.assertRaises(RuntimeError): c.step({}, .02)

    def test_no_hardware_services_or_command_destinations(self):
        path = Path(__file__).parents[1]/'tdfc_lab/observe.py'
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in
                             ('create_client', 'takeoff', 'land', 'cmdFullState') for n in calls))
        publishers = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'create_publisher']
        self.assertEqual(len(publishers), 1)
        self.assertIn('/tdfc/candidate/', ast.unparse(publishers[0]))

    def test_original_interface_validation_still_enforced(self):
        task = load_task('ring')
        c = CandidateController(task, self.names)
        p = np.array(task['hover_positions'])
        frame = dict(position=p, velocity=np.zeros((4,3)), measurement_time=0.,
                     own_velocities=np.zeros((4,3)), own_times=[0.]*4)
        out = c.step(frame, 0.)
        self.assertEqual(out['velocity'].shape, (4,3))
        self.assertTrue(np.isfinite(out['acceleration']).all())
        self.assertTrue(all(not i['safety_certificate_applicable'] for i in out['diagnostics']))
        with self.assertRaises(ValueError): c.step(frame, .04)


if __name__ == '__main__':
    unittest.main()
