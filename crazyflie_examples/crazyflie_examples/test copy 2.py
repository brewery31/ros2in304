"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np
import rclpy
import message_filters
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy.node


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.0

k = 0
c = 0
l  = []
have_land = False
def main():
    global k
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    
    Z=0.5
    allcfs = swarm.allcfs
    cflist = allcfs.crazyflies
    n = len(cflist)

    allcfs.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(1.5+Z)
    
    class TimeSyncNode(rclpy.node.Node):

        def __init__(self):
            super().__init__('sync_node')
            self.subscribers = []
            for i in range(n):
                cfid = int(cflist[i].uri[-2:], 16)
                self.subscribers.append(message_filters.Subscriber(self, PoseStamped, f"/cf{cfid}/pose"))
                
            queue_size = 10
            max_delay = 0.1
            self.time_sync = message_filters.ApproximateTimeSynchronizer(self.subscribers,
                                                        queue_size, max_delay)
            self.time_sync.registerCallback(self.SyncCallback)
            
        def SyncCallback(self, *args):
            global have_land
            global c
            if have_land:
                return
            if c > 30:
                # allcfs.land(targetHeight=0.08, duration=1.0 + Z)
                # timeHelper.sleep(1.0 + Z)
                have_land = True
                self.destroy_node()
                return

            # global l
            # l.append([args[0].pose.pose.position.y, args[1].pose.pose.position.y, args[2].pose.pose.position.y])
            # print(args[0].pose.pose.position.x)

            # if c == 10:
            #     print(*args)
            c += 1
            print(f'c{c}')
            # print(f'l{len(l)}')                
                
    # def call_back(*args):
    #     # pass
    #     global c
    #     global l
    #     l.append([args[0].pose.pose.position.y, args[1].pose.pose.position.y, args[2].pose.pose.position.y])
    #     # print(args[0].pose.pose.position.x)

    #     # if c == 10:
    #     #     print(*args)
    #     c += 1
    #     print(c)
    #     print(len(l))
    #     # print(timeHelper.time(), args[0].pose.pose.position.x)
    
    # def spin(*args):
    #     try:
    #         rclpy.spin(*args)
    #     except KeyboardInterrupt:
    #         print("here3")

    # node = rclpy.create_node('n111')
    node = TimeSyncNode()
    # subscribers = []
    # for i in range(n):
    #     cfid = int(cflist[i].uri[-2:], 16)
    #     subscribers.append(message_filters.Subscriber(node, Odometry, f"/cf{cfid}/odom"))
    #     # subscribers.append(message_filters.Subscriber(node, PoseStamped, f"/cf{cfid}/pose"))
    # sync = message_filters.ApproximateTimeSynchronizer(subscribers, 10, 0.1)
    # sync.registerCallback(call_back)
    
    print('here1')
    # # timeHelper.sleep(600)
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)

    thread.start()
    
    input()
    node.destroy_node()
    print("XXXXXXXXXXXXXXXXXX")

    # # print("here2")
    # # rclpy.shutdown()
    # # thread.join()
    # while rclpy.ok():
    #     print("XXXX")
    #     rclpy.spin(node) # no spin_once!
        # if (k%10 == 0):
        #     print(k)
        # break
        # k += 1
        # if k == 10:
        #     break
    # try:
    #     rclpy.spin(node)
    # except Exception as e:
    #     print(e)

    print("here2")
    print(k)
    print(c)
    print(l)
    print(len(l))
    # node.destroy_node()

    allcfs.land(targetHeight=0.08, duration=1.0 + Z)
    timeHelper.sleep(1.0 + Z)


if __name__ == '__main__':
    main()