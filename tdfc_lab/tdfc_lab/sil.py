"""In-process firmware SIL: no cflib, radio, aircraft topics or aircraft services."""
import csv
import json
from pathlib import Path
import numpy as np
from .mission import SimulationMission
from .scenarios import letter_targets,letter_edges


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Point, TransformStamped
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker, MarkerArray
    from tf2_ros import TransformBroadcaster
    from crazyflie_sim.crazyflie_sil import CrazyflieSIL
    from crazyflie_sim.sim_data_types import State
    from crazyflie_sim.backend.np import Backend

    class Simulator(Node):
        def __init__(self):
            super().__init__('tdfc_firmware_sim',namespace='tdfc_sim')
            self.declare_parameter('shape','ring');self.declare_parameter('agents',4)
            self.declare_parameter('seed',0);self.declare_parameter('formation_seconds',60.)
            self.declare_parameter('output_dir','tdfc_sil_run')
            self.mission = SimulationMission(self.get_parameter('shape').value,
                self.get_parameter('agents').value,seed=self.get_parameter('seed').value,
                formation_seconds=self.get_parameter('formation_seconds').value)
            self.output = Path(self.get_parameter('output_dir').value)
            self.output.mkdir(parents=True,exist_ok=False)
            self.states = [State(p) for p in self.mission.ground]
            self.names = self.mission.controller.names
            self.backend = Backend(self,self.names,self.states)
            self.cfs = [CrazyflieSIL(n,p,'mellinger',self.backend.time)
                        for n,p in zip(self.names,self.mission.ground)]
            for cf,state in zip(self.cfs,self.states):cf.setState(state)
            self.odom_publishers = [self.create_publisher(Odometry,f'{n}/odom',10) for n in self.names]
            self.marker_publisher = self.create_publisher(MarkerArray,'markers',1)
            self.tf = TransformBroadcaster(self)
            self.file = open(self.output/'trajectory.csv','x',newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['sim_time','phase','robot','x','y','z','vx','vy','vz'])
            self.minimum = float('inf');self.speed = 0.;self.tick_count = 0
            self.finished = False;self.status = 'interrupted';self.reason = None
            self.phase_seen = None
            self.timer = self.create_timer(.02,self.tick)
            self.get_logger().info('In-process SIM ONLY: firmware Mellinger + upstream NumPy rigid-body plant')

        def tick(self):
            if self.finished:return
            try:
                p = np.array([s.pos for s in self.states]);v = np.array([s.vel for s in self.states])
                t = self.tick_count*.02
                ref = self.mission.step(p,v,t)
                for i,cf in enumerate(self.cfs):
                    cf.cmdFullState(ref['position'][i],ref['velocity'][i],ref['acceleration'][i],0.,np.zeros(3))
                for _ in range(40):
                    desired = [cf.getSetpoint() for cf in self.cfs]
                    actions = [cf.executeController() for cf in self.cfs]
                    self.states = self.backend.step(desired,actions)
                    for cf,state in zip(self.cfs,self.states):cf.setState(state)
                    pp = np.array([s.pos for s in self.states])
                    vv = np.array([s.vel for s in self.states])
                    if not np.isfinite(np.r_[pp.ravel(),vv.ravel()]).all():raise RuntimeError('nonfinite rigid-body state')
                    dist=np.linalg.norm(pp[:,None]-pp[None],axis=2);np.fill_diagonal(dist,np.inf)
                    self.minimum=min(self.minimum,float(dist.min()))
                    self.speed=max(self.speed,float(np.linalg.norm(vv,axis=1).max()))
                    if dist.min()<.5:raise RuntimeError('sampled aircraft distance below 0.5m')
                self.tick_count+=1
                self.publish()
                if self.phase_seen != self.mission.phase:
                    self.phase_seen=self.mission.phase
                    self.get_logger().info(f't={self.tick_count*.02:.2f}s phase={self.phase_seen}')
                if self.mission.phase=='done':
                    self.status='completed_simulation';self.finished=True
                    self.timer.cancel();self.write_summary()
                    self.get_logger().info(f'Simulation complete; logs: {self.output}')
            except Exception as exc:
                self.status='failed_simulation';self.reason=str(exc);self.finished=True
                self.timer.cancel();self.write_summary();self.get_logger().error(str(exc))

        def stamp(self):
            ns=round(self.tick_count*.02*1e9)
            return Time(sec=ns//1000000000,nanosec=ns%1000000000)

        def publish(self):
            stamp=self.stamp();markers=MarkerArray()
            def point(x):return Point(x=float(x[0]),y=float(x[1]),z=float(x[2]))
            for i,(name,state,pub) in enumerate(zip(self.names,self.states,self.odom_publishers)):
                msg=Odometry();msg.header.frame_id='world';msg.header.stamp=stamp;msg.child_frame_id='sim_'+name
                for j,axis in enumerate('xyz'):
                    setattr(msg.pose.pose.position,axis,float(state.pos[j]))
                    setattr(msg.twist.twist.linear,axis,float(state.vel[j]))
                for axis,value in zip('wxyz',state.quat):setattr(msg.pose.pose.orientation,axis,float(value))
                pub.publish(msg)
                transform=TransformStamped();transform.header=msg.header;transform.child_frame_id='sim_'+name
                transform.transform.translation.x=float(state.pos[0]);transform.transform.translation.y=float(state.pos[1]);transform.transform.translation.z=float(state.pos[2])
                transform.transform.rotation=msg.pose.pose.orientation;self.tf.sendTransform(transform)
                m=Marker();m.header=msg.header;m.ns='aircraft';m.id=i;m.type=Marker.SPHERE;m.action=Marker.ADD
                m.pose=msg.pose.pose;m.scale.x=m.scale.y=m.scale.z=.12
                m.color.a=1.;m.color.r=.2+.7*(i%3==0);m.color.g=.2+.7*(i%3==1);m.color.b=.2+.7*(i%3==2)
                markers.markers.append(m)
                self.writer.writerow([self.tick_count*.02,self.mission.phase,name,*state.pos,*state.vel])
            guide=Marker();guide.header.frame_id='world';guide.header.stamp=stamp
            guide.ns='target_reference';guide.id=100;guide.pose.orientation.w=1.;guide.scale.x=.012
            guide.color.r=guide.color.g=guide.color.b=.5;guide.color.a=.8
            if self.mission.shape in ('R','A','L'):
                guide.type=Marker.LINE_LIST;goals=letter_targets(self.mission.shape)
                guide.points=[point(goals[i]) for edge in letter_edges(self.mission.shape) for i in edge]
            else:
                from .research.curve_catalog import catalog_parametric
                guide.type=Marker.LINE_STRIP
                goals=catalog_parametric(f'curve_demo_{self.mission.shape}_n{len(self.names)}',np.linspace(0,1,256))
                guide.points=[point(p) for p in goals]
            markers.markers.append(guide);self.marker_publisher.publish(markers)
            self.file.flush()

        def write_summary(self):
            (self.output/'summary.json').write_text(json.dumps(dict(status=self.status,reason=self.reason,
                shape=self.mission.shape,agents=len(self.names),method=self.mission.controller.method,
                simulated_seconds=self.tick_count*.02,phase=self.mission.phase,
                formation_terminal_metrics=self.mission.formation_summary,
                formation_success=(self.mission.formation_summary or {}).get('terminal_held_success'),
                min_distance_sampled_0p5ms_m=self.minimum if np.isfinite(self.minimum) else None,max_sampled_speed_mps=self.speed,
                model='upstream cffirmware Mellinger + crazyflie_sim.backend.np',
                full_state_reference='bounded velocity-loop predictor, not flight-calibrated',
                safety_certificate_applicable=False,hardware_flights=0),indent=2)+'\n')

        def destroy_node(self):
            self.write_summary();self.file.close();self.backend.shutdown()
            return super().destroy_node()

    rclpy.init(args=args);node=None
    try:
        node=Simulator();rclpy.spin(node)
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()
