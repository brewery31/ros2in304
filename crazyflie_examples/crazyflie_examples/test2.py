"""Takeoff-hover-land for one CF. Useful to validate hardware config."""


from crazyflie_py import Crazyswarm
import numpy as np
import rclpy
import message_filters
import threading
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from crazyflie_interfaces.msg import LogDataGeneric
import rclpy.node
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
import time
from crazyflie_interfaces.msg import TrajectoryPolynomialPiece, FullState, Position


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.0
Z = 0.5


def main():


    algorithm = 2
    simulation = False


    Z = 0.5
    sleepRate = 10
    dt = 1.0 / sleepRate
    goals = {int(i): {'pos': None, 'vel': None} for i in [0,1,2]}
    goals[0]['pos'] = np.array([-2.0, 0.0, Z])
    goals[1]['pos'] = np.array([2.0, 0.0, Z])
    # goals[2]['pos'] = np.array([2.0, 0.0, Z])
    # goals[2]['vel'] = np.array([0.0, 0.0, 0.0])
    goal = np.array([-2.0, 0.0, Z])
    alpha = 0.5
    if algorithm == 1:
        RADII = np.array([alpha/2, alpha/2, 0.4])
    else:
        RADII = np.array([0.15, 0.15, 0.4])
    
    def distance(p1, p2):
        return np.linalg.norm(p1-p2)
    def normalize(p):
        if np.linalg.norm(p)==0.0:
            return p
        return p/(np.linalg.norm(p))
    
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    leader = 5
    allcfs = swarm.allcfs
    cflist = allcfs.crazyflies
    n = len(cflist)
    last_target_position = None
    print(n)
    #cfs = {i: {'pos': None, 'vel': None} for i in range(n)}
    cfs = {int(cflist[i].uri[-2:], 16): {'pos': None, 'vel': None, 'dis':float('inf')}
           for i in range(len(cflist))}
    allcfs.takeoff(targetHeight=Z, duration=2 + Z)
    timeHelper.sleep(2 + Z)
    

    pltpoints = [[[] for j in range(3)] for i in range(n)]
    
    class TimeSyncNode(rclpy.node.Node):


        def __init__(self):
            super().__init__("sync_node")
            # self.publisherhub = []
            self.subscribers = []
            # self.cmdPositionMsg = Position()
            for i in range(n):
                cfid = int(cflist[i].uri[-2:], 16)
                print(cfid)
                self.subscribers.append(
                    # message_filters.Subscriber(self, PoseStamped, f"/cf{cfid}/pose")
                    # message_filters.Subscriber(self, LogDataGeneric, f"/cf{cfid}/tpc1")
                    message_filters.Subscriber(self, Odometry, f"/cf{cfid}/odom")
                )
            queue_size = 10
            max_delay = 0.01
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, queue_size, max_delay
            )
            # timeHelper.sleep(10)
            # print('1')
            self.time_sync.registerCallback(self.SyncCallback)


            # self.startTime = timeHelper.time()
            # self.radius = 0.5
            # self.roundTime = 10.0
            # self.kPosition = 1.0
            
            self.count = 0
            self.SHOULD_LAND = False
            self.LANDING = False


        def SyncCallback(self, *args):
            print('Y')
            # print(goals)
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
            
            for i, id in enumerate(cfs.keys()):
                #pos = np.array(
                #    [args[i].pose.position.x,
                #    args[i].pose.position.y,
                #    args[i].pose.position.z]
                #)
                pos = np.array(
                     [args[i].pose.pose.position.x,
                    args[i].pose.pose.position.y,
                    Z]
                )
                vel = np.array(
                    [args[i].twist.twist.linear.x,
                    args[i].twist.twist.linear.y,
                    0]
                )
                #pos = np.array(
                #    [args[i].values[0],
                #    args[i].values[1],
                #    Z]                # obstacles = []
                #)
                #vel = np.array(
                #    [args[i].values[2],
                #    args[i].values[3],
                #    0]
                #)
                cfs[id]['pos'] = pos
                cfs[id]['vel'] = vel
                #cfs[id]['vel'] = np.zeros(3)
            print(cfs)


            # last_p = None
            if distance(cfs[leader]['pos'], goals[0]['pos']) > 0.1:
                # if last_p is None:
                #     last_p = p = [cfs[id]['pos'] for id in cfs.keys()]
                # else:
                #     last_p = p
                p = [cfs[id]['pos'] for id in cfs.keys()]
                if simulation:
                    v = [cfs[id]['vel'] for id in cfs.keys()]
                else:
                    # v = [(p[i]-last_p[i])/dt for i in range(n)]
                    v = [cfs[id]['vel'] for id in cfs.keys()]
                # if distance(cfs[leader]['pos'], goals[2]['pos']) < 0.1:
                for i in range(n):
                    for j in range(3):
                        pltpoints[i][j].append(p[i][j])
                if algorithm == 1:
                    for i in range(n):
                        if int(i) == int(leader):
                            target_position = goal
                        else:
                            target_position = p[0]+alpha*normalize(p[i]-p[0])
                        if simulation:
                            cflist[i].cmdVelocityWorld(0.2*normalize(target_position-p[i]), yawRate=1.0)
                        else:
                            print(target_position)
                            cflist[i].cmdPosition(target_position, yaw=0)
                            # v = (target_position-pos)/30
                            # cflist[i].cmdFullState(pos = target_position, vel= v, acc = np.zeros(3), yaw=0.0, omega=np.zeros(3)) 
                else:
                    for i in range(n):
                        di = np.zeros(3)
                        if any(distance(p[i], p[j])<alpha for j in range(n) if i!=j):
                            for j in range(n):
                                if j!=i and distance(p[i], p[j])<alpha:
                                    di -= normalize(p[j]-p[i])
                        else:
                            for j in range(n):
                                if j!=i:
                                    di += normalize(p[j]-p[i])
                                # di += normalize(v[j])
                        di = normalize(di)
                        if  int(cflist[i].uri[-2:], 16)== leader:
                            di = normalize(di+2*normalize(goals[0]['pos']-p[i]))
                            # goals[2]['pos'] = p[i] + di * 0.2
                            # goals[2]['vel'] = (goals[2]['pos']- p[i])/5
                        print(i, di)
                        target_position = p[i] + di * 0.2
                        target_position[2] = Z  # 确保高度保持在Z
                        if simulation:
                            cflist[i].cmdVelocityWorld(di*0.2, yawRate=1.0)
                        else:
                            print(target_position)
                            # cflist[i].cmdPosition(target_position, yaw=0)
                            # vt = (target_position-p[i])*30
                            vt = (target_position-p[i])
                            a=(vt-v[i])
                            # print(vt, a)
                            # if self.count % 10 == 0:
                                # if self.count<10:
                                #     cflist[i].cmdFullState(pos=target_position, vel=vt*0.2, acc=a*0.2, yaw=0.0,  omega=np.zeros(3))
                            cflist[i].cmdFullState(pos=target_position, vel=vt/10, acc=a, yaw=0.0,  omega=np.zeros(3))
                            timeHelper.sleepForRate(sleepRate)
                # else:
                #     target_position = goals[2]['pos']
                #     vt = goals[2]['vel']
                #     a = (vt-v[i])
                #     cflist[i].cmdFullState(pos=target_position, vel=vt, acc=a, yaw=0.0,  omega=np.zeros(3))
            else:
                if (goals[0]['pos']-goals[1]['pos']).any():
                    goals[0] = goals[1]
                else:
                    for cf in allcfs.crazyflies:
                        cf.notifySetpointsStop()
                        cf.land(targetHeight=0.08, duration=1.0 + Z)
                    timeHelper.sleep(2.0 + Z)
                    allcfs.emergency()
                    node.destroy_node()
            self.count += 1
            if self.count % 2 == 0:
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


    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.08, duration=1.0 + Z)
    timeHelper.sleep(2.0 + Z)


    ax = plt.figure().add_subplot(projection='3d')
    for i in range(n):
        ax.plot(*pltpoints[i], label=f'cf{i+1}')
    ax.legend()
    ax.set_xlim([-1, goal[0]])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    plt.show()


if __name__ == "__main__":
    main()