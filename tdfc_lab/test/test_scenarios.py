import ast
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from tdfc_lab.adapter import CandidateController,StateBuffer,load_task
from tdfc_lab.scenarios import validate_scene,hover_grid,letter_targets,SceneController
from tdfc_lab.mission import polynomial_reference,full_state_reference,SimulationMission
from tdfc_lab.offline import replay


class ScenarioTests(unittest.TestCase):
    def test_supported_matrix_and_rejection(self):
        for s,n in [('ring',4),('ring',6),('helix',4),('helix',6),('R',5),('A',5),('L',5)]:
            validate_scene(s,n)
            scene=SceneController(s,n)
            self.assertEqual(len(scene.names),n)
            p=hover_grid(n)
            u=scene.command(p,np.zeros_like(p),0.)
            self.assertEqual(u.shape,(n,3))
            self.assertLessEqual(np.linalg.norm(u,axis=1).max(),.3+1e-9)
        for s,n in [('ring',5),('helix',8),('R',4),('A',6),('L',6),('other',5)]:
            with self.assertRaises(ValueError):validate_scene(s,n)

    def test_goals_distinct_and_safe_grid_and_6th_name(self):
        for n in (4,5,6):
            p=hover_grid(n,3);d=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(d,np.inf)
            self.assertGreater(d.min(),.85)
            StateBuffer(tuple(f'cf{i}' for i in range(n)))
        self.assertEqual(SceneController('ring',6).names[-1],'cfA')
        for s in ('R','A','L'):
            p=letter_targets(s);d=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(d,np.inf)
            self.assertGreaterEqual(d.min(),.7-1e-12)
            self.assertGreaterEqual(p[:,2].min(),.6-1e-12)

    def test_optional_shaping_cannot_modify_governor_output(self):
        task=copy.deepcopy(load_task('ring'));task['sim'].update(tangential_damping=.7,command_filter_tau=.2)
        names=('a','b','c','d');controller=CandidateController(task,names)
        expected=np.tile([.03,.02,.01],(4,1))
        p=hover_grid(4);v=np.zeros_like(p)
        with patch('tdfc_lab.research.curve_ego_interface.DirectionalGovernor.command',return_value=(expected,{'speed_limit':.3})):
            out=controller.step(dict(position=p,velocity=v,measurement_time=0.,own_velocities=v,own_times=[0.]*4),0.)
        np.testing.assert_array_equal(out['velocity'],expected)

    def test_polynomial_endpoints_and_reference_acceleration_bound(self):
        p=hover_grid(6);ground=p.copy();ground[:,2]=0
        first=polynomial_reference(ground,p,0.)
        last=polynomial_reference(ground,p,10.)
        np.testing.assert_array_equal(first['position'],ground)
        np.testing.assert_allclose(last['position'],p)
        np.testing.assert_allclose(last['velocity'],0)
        speeds=[np.linalg.norm(polynomial_reference(ground,p,t)['velocity'],axis=1).max() for t in np.linspace(0,10,101)]
        self.assertLess(max(speeds),.3)
        ref=full_state_reference(p,np.zeros_like(p),np.ones_like(p)*10)
        self.assertLessEqual(np.linalg.norm(ref['acceleration'],axis=1).max(),1.+1e-9)
        self.assertLessEqual(np.linalg.norm(ref['velocity'],axis=1).max(),.27+1e-9)

    def test_takeoff_requires_feedback_and_return_precedes_land(self):
        mission=SimulationMission('ring',4,formation_seconds=.04)
        p=mission.home.copy();v=np.zeros_like(p)
        for k in range(551):mission.step(p,v,k*.02)
        self.assertEqual(mission.phase,'formation')
        for k in range(551,555):mission.step(p,v,k*.02)
        self.assertEqual(mission.phase,'return')
        for k in range(555,610):mission.step(p,v,k*.02)
        self.assertEqual(mission.phase,'land')
        for k in range(610,1115):mission.step(mission.ground,v,k*.02)
        self.assertEqual(mission.phase,'done')

    def test_replay_rejects_nan_expectations_and_real_mismatch(self):
        fixture=Path(__file__).parent/'fixtures/ring.npz'
        with np.load(fixture) as z:arrays={k:z[k][:2] if k!='agent_ids' else z[k] for k in z.files}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invalid.npz'
            arrays['expected_command']=arrays['expected_command'].copy()
            arrays['expected_command'][0,0,0]=np.nan
            np.savez_compressed(p,**arrays)
            with self.assertRaises(ValueError):replay(p,load_task('ring'))
            arrays['expected_command'][0,0,0]=1.
            np.savez_compressed(p,**arrays)
            with self.assertRaises(AssertionError):replay(p,load_task('ring'))

    def test_simulation_entry_has_no_aircraft_command_transport(self):
        tree=ast.parse((Path(__file__).parents[1]/'tdfc_lab/sil.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('create_client','create_subscription') for n in calls))
        imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('cflib' in n or 'crazyflie_py' in n for n in imports))


if __name__=='__main__':unittest.main()
