#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


def executeTrajectory(timeHelper, cflist, trajpath, rate=100, offset=np.zeros(3)):
    traj = Trajectory()
    traj.loadcsv(trajpath)

    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > traj.duration:
            break

        e = traj.eval(t)
        for cf in cflist:
            cf.cmdFullState(
                e.pos + np.array(cf.initialPosition) + offset,
                e.vel,
                e.acc,
                e.yaw,
                e.omega)

        timeHelper.sleepForRate(rate)


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cflist = swarm.allcfs.crazyflies

    rate = 20.0
    Z = 0.8

    for cf in cflist:
        cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    executeTrajectory(timeHelper, cflist,
                      Path(__file__).parent / 'data/figure8.csv',
                      rate,
                      offset=np.array([0, 0, Z]))

    for cf in cflist:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.08, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == '__main__':
    main()
