#!/usr/bin/env python

# killall -s SIGKILL python
import numpy as np
from crazyflie_py import Crazyswarm
import matplotlib.pyplot as plt
import rclpy
import message_filters
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def main():
    algorithm = 2
    simulation = 1

    Z = 0.5
    sleepRate = 10
    dt = 1.0 / sleepRate
    goal = np.array([3.0, 0.0, Z])
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
    allcfs = swarm.allcfs
    cflist = allcfs.crazyflies

    n = len(cflist)
    leader = cflist[0]
    follower = cflist[1:]
    
    def call_back(**kwargs):
        print(kwargs)
        print(3)
        timeHelper.sleep(600)
        raise Exception
        
    node = rclpy.create_node('node')
    subscribers = []
    for i in range(n):
        cfid = int(cflist[i].uri[-2:], 16)
        subscribers.append(message_filters.Subscriber(node, Odometry, f"/cf{cfid}/odom"))
    sync = message_filters.ApproximateTimeSynchronizer(subscribers, 10, 0.1, allow_headerless=True)
    sync.registerCallback(call_back)

    rate = node.create_rate(10)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
        print('here')
        rate.sleep()
        print('here')
    
    # for i in range(n):
    #     others = cflist[:i] + cflist[i+1:]
    #     cflist[i].enableCollisionAvoidance(others, RADII)
    while True:
        pass

    allcfs.takeoff(targetHeight=Z, duration=3.0+Z)
    timeHelper.sleep(4.0+Z)
    
    pltpoints = [[[] for j in range(3)] for i in range(n)]
    last_p = None
    
    position = {}
    while distance(leader.position(), goal) > 0.1:
        if last_p is None:
            last_p = p = [cflist[i].position() for i in range(n)]
        else:
            last_p = p
            p = [cflist[i].position() for i in range(n)]
        if simulation:
            v = [cflist[i].velocity() for i in range(n)]
        else:
            v = [(p[i]-last_p[i])/dt for i in range(n)]
        for i in range(n):
            for j in range(3):
                pltpoints[i][j].append(p[i][j])

        if algorithm == 1:
            for i in range(n):
                if cflist[i] is leader:
                    target_position = goal
                else:
                    target_position = p[0]+alpha*normalize(p[i]-p[0])
                if simulation:
                    cflist[i].cmdVelocityWorld(0.2*normalize(target_position-p[i]), yawRate=1.0)
                else:
                    cflist[i].cmdPosition(target_position, yaw=0)
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
                        di += normalize(v[j])
                di = normalize(di)
                if cflist[i] is leader:
                    di = normalize(di+2*normalize(goal-p[i]))
                target_position = p[i] + di * 0.2
                # target_position[2] = Z  # 确保高度保持在Z
                if simulation:
                    cflist[i].cmdVelocityWorld(di*0.2, yawRate=1.0)
                else:
                    cflist[i].cmdPosition(target_position, yaw=0)
        timeHelper.sleepForRate(sleepRate)
    allcfs.land(targetHeight=0.1, duration=3.0+Z)
    timeHelper.sleep(3.0+Z)


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
