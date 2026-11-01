"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from crazyflie_py import Crazyswarm
import numpy as np

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.0
Z = 1.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    
    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    # i=0
    # while(i<3):
    # # cf.goTo(np.array([1.0, 1.0, 1.0]), 0, 2.0)
    #     cf.goTo(np.array([1.0, 0.0, Z]), 0, 2.0)
    #     timeHelper.sleep(2)
        
    #     cf.goTo(np.array([1.0, -1.0, Z]), 0, 2.0)
    #     timeHelper.sleep(2)
        
    #     # cf.goTo(np.array([0.0, 0.0, Z]), 0, 2.0)
    #     # timeHelper.sleep(2)
        
    #     i+=1
    # # cf.land(targetHeight=0.08, duration=2.5)
    cf.land(targetHeight=0.08, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == '__main__':
    main()
