"""Observation-only ROS node. No aircraft command topics or services are used."""
import json
import math
import threading
import numpy as np
from .adapter import StateBuffer, CandidateController, load_task


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from crazyflie_interfaces.msg import FullState

    class Observer(Node):
        def __init__(self):
            super().__init__('tdfc_observation_only')
            self.declare_parameter('shape', 'ring')
            self.declare_parameter('robot_names', ['cf5', 'cf6', 'cf7', 'cf8'])
            self.declare_parameter('expected_frame', 'world')
            self.declare_parameter('log_path', 'tdfc_candidate.jsonl')
            names = tuple(self.get_parameter('robot_names').value)
            self.frame = self.get_parameter('expected_frame').value
            self.buffer = StateBuffer(names)
            self.task = load_task(self.get_parameter('shape').value, agents=len(names))
            self.controller = CandidateController(self.task, names)
            self.started = None
            self.last_tick = -1
            self.fault = False
            self.lock = threading.Lock()
            self.file = open(self.get_parameter('log_path').value, 'x', buffering=1)
            self.subscribers = [self.create_subscription(Odometry, f'/{n}/odom',
                                lambda msg, n=n: self.receive(n, msg), 10) for n in names]
            self.candidate_publishers = [self.create_publisher(FullState,
                              f'/tdfc/candidate/{n}/cmd_full_state', 1) for n in names]
            self.timer = self.create_timer(.02, self.tick)
            self.get_logger().warning('OBSERVATION ONLY: uncalibrated candidate topics; no takeoff or hardware send')

        def receive(self, name, msg):
            if self.fault:
                return
            try:
                if msg.header.frame_id.lstrip('/') != self.frame.lstrip('/'):
                    raise ValueError('odom frame differs from explicitly configured world frame')
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
                p, v = msg.pose.pose.position, msg.twist.twist.linear
                with self.lock:
                    self.buffer.add(name, stamp, [p.x, p.y, p.z], [v.x, v.y, v.z])
            except Exception as e:
                self.fail(e)

        def fail(self, error):
            self.fault = True
            self.file.write(json.dumps(dict(event='latched_fault', reason=str(error)))+'\n')
            self.get_logger().error(f'Candidate generation stopped: {error}')

        def tick(self):
            if self.fault:
                return
            now = self.get_clock().now().nanoseconds*1e-9
            if self.started is None:
                try:
                    self.buffer.aligned(now)
                except ValueError:
                    return
                self.started = now
            tick = math.floor((now-self.started+1e-7)/.02)
            if tick == self.last_tick:
                return
            try:
                if tick != self.last_tick+1:
                    raise ValueError('missed decision tick or clock jump')
                decision = self.started + tick*.02
                if abs(now-decision) > .005:
                    raise ValueError('decision scheduling jitter exceeds draft 5ms guard')
                with self.lock:
                    frame = self.buffer.aligned(decision)
                out = self.controller.step(frame, decision)
                self.last_tick = tick
                for i, publisher in enumerate(self.candidate_publishers):
                    msg = FullState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = self.frame
                    msg.pose.orientation.w = 1.
                    for axis, j in zip('xyz', range(3)):
                        setattr(msg.pose.position, axis, float(out['position'][i,j]))
                        setattr(msg.twist.linear, axis, float(out['velocity'][i,j]))
                        setattr(msg.acc, axis, float(out['acceleration'][i,j]))
                    publisher.publish(msg)
                self.file.write(json.dumps(dict(event='candidate', decision_time=decision,
                    measurement_time=frame['measurement_time'], names=self.buffer.names,
                    velocity=out['velocity'].tolist(), position=out['position'].tolist(),
                    acceleration=out['acceleration'].tolist(), safety_certificate_applicable=False,
                    timestamp_semantics='receive_time_not_measurement_time'))+'\n')
            except Exception as e:
                self.fail(e)

        def destroy_node(self):
            self.file.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = Observer()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
