"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np
import rclpy
import message_filters
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy.node
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
import time

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.0
Z = 0.5

def main():

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    allcfs = swarm.allcfs
    cflist = allcfs.crazyflies
    n = len(cflist)

    allcfs.takeoff(targetHeight=Z, duration=2 + Z)
    timeHelper.sleep(2 + Z)

    class TimeSyncNode(rclpy.node.Node):

        def __init__(self):
            super().__init__("sync_node")
            self.subscribers = []
            for i in range(n):
                cfid = int(cflist[i].uri[-2:], 16)
                print(cfid)
                self.subscribers.append(
                    message_filters.Subscriber(self, PoseStamped, f"/cf{cfid}/pose")
                )
            queue_size = 10
            max_delay = 0.01
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, queue_size, max_delay
            )
            self.time_sync.registerCallback(self.SyncCallback)

            self.startTime = timeHelper.time()
            self.radius = 0.5
            self.roundTime = 10.0
            self.kPosition = 1.0
            
            self.count = 0
            self.SHOULD_LAND = False
            self.LANDING = False


        def SyncCallback(self, *args):
            if self.SHOULD_LAND:
                print("here1")
                if self.LANDING:
                    return
                print("here2")
                self.LANDING = True
                req = NotifySetpointsStop.Request()
                self.notify_client.call_async(req)
                req = Land.Request()
                req.height = 0.1
                req.duration = rclpy.duration.Duration(seconds=2.0).to_msg()
                self.land_client.call_async(req)
                time.sleep(2.0)
                print("here3")
                return
            for i in range(n):
                pos = np.array(
                    [args[i].pose.position.x,
                    args[i].pose.position.y,
                    args[i].pose.position.z]
                )
                # print(pos)
                startPos = cflist[i].initialPosition + np.array([0, 0, Z])
                center_circle = startPos - np.array([self.radius, 0, 0])
                time = timeHelper.time() - self.startTime
                omega = 2 * np.pi / self.roundTime
                vx = -self.radius * omega * np.sin(omega * time)
                vy = self.radius * omega * np.cos(omega * time)
                desiredPos = center_circle + self.radius * np.array(
                    [np.cos(omega * time), np.sin(omega * time), 0]
                )
                cflist[i].cmdPosition(pos = desiredPos)
                # errorX = desiredPos - pos
                # cflist[i].cmdVelocityWorld(
                #     np.array([vx, vy, 0] + self.kPosition * errorX), yawRate=0
                # )
                # print(desiredPos)
            self.count += 1
            if self.count % 10 == 0:
                print(self.count)


    node = TimeSyncNode()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    
    s = input()
    if s == '':
        print("XXXXXXXXXXXXXXXXXX")
        allcfs.emergency()
        node.destroy_node()
    else:
        print("SHOULD_LAND")
        node.SHOULD_LAND = True
        timeHelper.sleep(100)



if __name__ == "__main__":
    main()
