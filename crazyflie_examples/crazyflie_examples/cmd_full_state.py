#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
import numpy as np
import json


n = 10
ignore_agents = {
    # 'agent1',
    # 'agent2',
    # 'agent3',
    # 'agent4',
    # 'agent5',
    # 'agent6',
    # 'agent7',
    # 'agent8',
    # 'agent9',
    # 'agent10',
}
rate = 10.0
Z = 0.5

real_pos_min = [-1.0, -1.0, Z]
real_pos_max = [2.0, 1.0, None]
data_pos_min = [None, None, None]
data_pos_max = [None, None, None]


def main():

    with open(
        "/home/crazyswarm2/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/data/simtoreal_10uavs/final_positions_0_100_Num10_0.json",
        "r",
    ) as fp:
        data = json.load(fp)
        data = data["group_0"]
        for dim in [0, 1]:
            data_pos_min[dim] = min(agent_data[0]["position"][dim] for agent_data in data.values())
            data_pos_max[dim] = max(agent_data[0]["position"][dim] for agent_data in data.values())
        for dim in [2]:
            data_pos_min[dim] = min(frame_data["position"][dim] for agent_data in data.values() for frame_data in agent_data)
            data_pos_max[dim] = max(frame_data["position"][dim] for agent_data in data.values() for frame_data in agent_data)
        average_coef = sum(
            (real_pos_max[dim] - real_pos_min[dim])
            / (data_pos_max[dim] - data_pos_min[dim])
            for dim in [0, 1]
        )
        average_coef /= 2
        for dim in [2]:
            real_pos_max[dim] = (
                real_pos_min[dim] + (data_pos_max[dim] - data_pos_min[dim]) * average_coef
            )
        for agent_data in data.values():
            for frame_data in agent_data:
                for dim in [0, 1, 2]:
                    frame_data["position"][dim] = real_pos_min[dim] + (
                        real_pos_max[dim] - real_pos_min[dim]
                    ) * (
                        (frame_data["position"][dim] - data_pos_min[dim])
                        / (data_pos_max[dim] - data_pos_min[dim])
                    )
        data = {agent:data[agent] for agent in data if agent not in ignore_agents}
        assert n == len(data)
        

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cflist = swarm.allcfs.crazyflies

    for cf in cflist:
        cf.takeoff(targetHeight=Z, duration=Z + 1.0)
    timeHelper.sleep(Z + 2.0)

    last_pos_list = []
    last_vel_list = []

    for cf, agent_data in zip(cflist, data.values()):
        pos = np.array(agent_data[0]["position"])
        cf.goTo(pos, 0, 1.0)
        
        last_pos_list.append(pos)
        last_vel_list.append(np.array([0.0, 0.0, 0.0]))
    timeHelper.sleep(1.5)

    for frame in range(len(next(iter(data.values())))):
        for cf, agent_data, last_pos, last_vel in zip(cflist, data.values(), last_pos_list, last_vel_list):
            pos = np.array(agent_data[frame]["position"])
            vel = (pos - last_pos)/rate # 为什么*rate不行
            acc = (vel - last_vel)/rate
            cf.cmdFullState(
                pos,
                vel,
                acc,
                yaw=0.0,
                omega=np.zeros(3)
            )
            last_pos = pos
            last_vel = vel

        timeHelper.sleepForRate(rate)


    for cf in cflist:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.15, duration=Z + 2.0)
    timeHelper.sleep(Z + 3.0)

    # 手动降落，还有bug

    # land_duration = Z + 2.0
    # land_targetHeight = 0.15
    
    # for cf, pos, vel in zip(cflist, last_pos_list, last_vel_list):
    #     vel = np.array([0.0, 0.0, (land_targetHeight-pos[2])/land_duration])
    
    # for frame in range(int(land_duration * rate)):
    #     for cf, pos, vel in zip(cflist, last_pos_list, last_vel_list):
    #         pos += vel / rate
    #         acc = np.zeros(3)
    #         cf.cmdFullState(
    #             pos,
    #             vel,
    #             acc,
    #             yaw=0.0,
    #             omega=np.zeros(3)
    #         )
    #         last_pos = pos
    #         last_vel = vel

if __name__ == "__main__":
    main()
