import unittest
from unittest.mock import patch
import numpy as np
from tdfc_lab.research.curve_ego_interface import CurveEgoInterface
from tdfc_lab.research.curve_tasks import curve_acceleration
from tdfc_lab.research.sim_protocol import clip_norm
from tdfc_lab.research.orca_adapter import orca_velocity
from tdfc_lab.research.directional_governor import DirectionalGovernor


class CurveEgoInterfaceTests(unittest.TestCase):
    def task(self):
        return dict(shape='curve_demo_ring_n4',n_agents=4,method='curve_ours',research_mode='observation_stress',params={},
                    observation_processor=dict(smoothing_seconds=0.,predict_to_now=True),curve_controller=dict(endpoint_cone=True),
                    sim=dict(execution_model='physical_zoh',stopping_governor=True,governor_geometry='directional',
                             orca_nonfinite_policy='ego_brake',a=.8,d_safe=.5,v_max=.3,a_max=1.,margin=.08,nominal_a_max=.8,actuator_tau=.1))

    def frame(self):
        return dict(agent_ids=['a','b','c','d'],position=np.array([[-.7,-.6,1.2],[.7,-.6,1.2],[.7,.6,1.2],[-.7,.6,1.2]]),
                    velocity=np.zeros((4,3)),measurement_time=0.,decision_time=0.,own_velocity=np.array([.03,0,0]),own_velocity_time=0.)

    def interface(self):return CurveEgoInterface(self.task(),ego_id='a',agent_ids=['a','b','c','d'])

    def test_ego_output_matches_joint_math_without_other_actual_velocities(self):
        obj=self.interface();frame=self.frame();c=obj.cfg
        acc,_=curve_acceleration(frame['position'],frame['velocity'],obj.cloud,obj.prm,.8,endpoint_cone=True)
        actual=np.array([[.03,0,0],[.08,.01,0],[-.02,.07,0],[0,-.05,.01]])
        preferred=clip_norm(actual+c.dt/(-np.expm1(-c.dt/c.actuator_tau))*acc,c.v_max)
        selected,_=orca_velocity(frame['position'],frame['velocity'],preferred,dt=c.dt,r_vis=obj.prm.r_vis_abs,d_safe=c.d_safe,v_max=c.v_max,horizon=c.horizon,margin=c.margin,nonfinite_policy='ego_brake')
        expected,_=DirectionalGovernor().command(frame['position'],frame['velocity'],actual,selected,dt=c.dt,a_max=c.a_max,tau=c.actuator_tau,v_max=c.v_max,r_vis=obj.prm.r_vis_abs,d_safe=c.d_safe,margin=c.margin)
        result,info=obj.step(**frame)
        np.testing.assert_allclose(result,expected[0],rtol=0,atol=1e-12)
        self.assertEqual(result.shape,(3,));self.assertFalse(info['safety_certificate_applicable'])

    def test_reordered_missing_and_invalid_frames_raise_before_output(self):
        for updates in [dict(agent_ids=['b','a','c','d']),dict(agent_ids=['a','b','c']),dict(coordinate_frame='NED'),
                        dict(measurement_time=.1),dict(measurement_time=-.2),dict(own_velocity_time=-.1),
                        dict(own_velocity=np.array([np.nan,0,0])),dict(own_velocity=np.array([.4,0,0]))]:
            with self.assertRaises(ValueError):self.interface().step(**(self.frame()|updates))

    def test_failed_input_does_not_consume_the_valid_first_frame(self):
        obj=self.interface();frame=self.frame()
        with self.assertRaises(ValueError):obj.step(**(frame|dict(coordinate_frame='NED')))
        first,_=obj.step(**frame);other,_=self.interface().step(**frame)
        np.testing.assert_array_equal(first,other)

    def test_non_nominal_cadence_is_not_silently_accepted(self):
        obj=self.interface();obj.step(**self.frame())
        with self.assertRaises(ValueError):obj.step(**(self.frame()|dict(decision_time=.04,measurement_time=.04,own_velocity_time=.04)))

    def test_only_selected_shapes_and_processor_are_supported(self):
        for change in [dict(shape='curve_demo_saddle_n4'),dict(observation_processor=dict(smoothing_seconds=.1,predict_to_now=True)),dict(research_mode='')]:
            with self.assertRaises(ValueError):CurveEgoInterface(self.task()|change,ego_id='a',agent_ids=['a','b','c','d'])

    def test_orca_runtime_failure_is_loud_not_an_unchecked_zero_command(self):
        with patch('tdfc_lab.research.curve_ego_interface.orca_velocity',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):self.interface().step(**self.frame())


if __name__=='__main__':unittest.main()
