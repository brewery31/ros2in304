import math
import time
from crazyflie_py import Crazyswarm
import numpy as np
import rclpy
import message_filters
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from crazyflie_interfaces.msg import LogDataGeneric
from itertools import combinations, product
from random import choice


TAKEOFF_DURATION = 2.0
Z = 0.7
f = 60.0

# config parameters
N = 3
N_top = 2
r = 0.1 # not used
u_ref = np.array([-1.0, 0, 0])
v_ref = 0.8
d_ref = 0.8 # essential
a_max = 2
v_max = 1

mass = 1
dt = 1/f

# optimzied parameters

W_sep = 7.83
W_coh = 7.32
W_alig = 9.71

W_obs = 3.97
r_0 = 0.07 # with obstacles
v_obs = 1.46
p_obs = 3.91
a_obs = 3.80
W_mig = 4.47

# adjust
# W_sep = 17.83
# W_coh = 3.32
# W_mig = 20
# W_obs = 1.97
W_alig = 1.71
d_ref = 0.8



vertices = list(product([1.5,-1.5], [1.5,-1.5], [Z, 1.5]))
temp = 1.0
vertices = list(product([temp, -temp], [temp, -temp], [Z, 1.4]))
# vertices = list(product([1.7, -1.7], [0], [0.6, 1.5]))+list(product([0], [1.7, -1.7], [0.6, 1.5]))
# vertices = list(product([2, -2], [0], [0.6, 1.5]))+list(product([0],[2, -2], [0.6, 1.5]))
# vertices = list(product([2], [0], [1, 2]))
vertices = [np.array(j) for j in vertices]

# vertices = [np.array(j+(Z,)) for j in vertices]

def D(r, a, p):
    if r <= 0:
        return 0
    if r*p < a/p:
        return r*p
    else:
        return (2*a*r-a**2/p**2)**0.5
    
def main():
    file_name = 'log2'

    np.set_printoptions(precision=3)
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    allcfs = swarm.allcfs
    cflist = allcfs.crazyflies
    n = len(cflist)
    temp = choice(list(range(len(vertices))))
    temp = 7
    print(f"goal: {temp}")
    cfs = {int(cflist[i].uri[-2:], 16): {'statep': None, 'statev': None, 
                                         'goal_idx': temp, 'goal_dir': np.array([-1.0, 0, 0]), 
                                         'setpointp': None, 'setpointv': None, 'dis':float('inf')}
           for i in range(len(cflist))}
    

    allcfs.takeoff(targetHeight=Z, duration=TAKEOFF_DURATION+Z)
    timeHelper.sleep(TAKEOFF_DURATION+Z+1)

    class TimeSyncNode(rclpy.node.Node):

        def __init__(self):
            super().__init__("sync_node")
            self.subscribers = []
            for i in range(n):
                cfid = int(cflist[i].uri[-2:], 16)
                self.subscribers.append(
                    message_filters.Subscriber(self, Odometry, f"/cf{cfid}/odom")
                    # message_filters.Subscriber(self, PoseStamped, f"/cf{cfid}/pose")
                    # message_filters.Subscriber(self, LogDataGeneric, f"/cf{cfid}/tpc1")
                )
            queue_size = 10
            max_delay = 1/f
            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, queue_size, max_delay
            )
            self.time_sync.registerCallback(self.SyncCallback)
            
            self.count = 0
            self.t = 0
            self.dis = []


        def SyncCallback(self, *args):


            for i, id in enumerate(cfs.keys()):
                # pos = np.array(
                #     [args[i].pose.pose.position.x,
                #     args[i].pose.pose.position.y,
                #     Z]
                # )

                # vel = np.array(
                #     [args[i].twist.twist.linear.x,
                #     args[i].twist.twist.linear.y,
                #     0]
                # )
                pos = np.array(
                    [args[i].pose.pose.position.x,
                    args[i].pose.pose.position.y,
                    args[i].pose.pose.position.z]
                )

                vel = np.array(
                    [args[i].twist.twist.linear.x,
                    args[i].twist.twist.linear.y,
                    args[i].twist.twist.linear.z]
                )
                # pos = np.array(
                #     [args[i].values[0],
                #     args[i].values[1],
                #     Z]
                # )
                # vel = np.array(
                #     [args[i].values[2],
                #     args[i].values[3],
                #     0]
                # )
                # pos = np.array(
                #     [args[i].pose.position.x,
                #     args[i].pose.position.y,
                #     args[i].pose.position.z]
                # )

                cfs[id]['statep'] = pos
                cfs[id]['statev'] = vel
                
                if cfs[id]['setpointp'] is None:
                    cfs[id]['setpointp'] = pos
                    cfs[id]['setpointv'] = vel
            
            
            for i, j in combinations(cfs.keys(), 2):
                dis = np.linalg.norm(cfs[i]['statep']-cfs[j]['statep'])
                if  dis < 0.3:
                    self.dis.append(f"{i} {j} {dis}")
                       
            if self.count == 0:
                with open(file=file_name, mode='w+') as f:
                        f.write(f'\n')

            
            
            # update                # v = vi + a*dt
                # v = v/np.linalg.norm(v)*min(np.linalg.norm(v), v_max)
                # pos = pi + (vi+v)/2*dt
            for i in cfs.keys():
                pi = cfs[i]['statep'] # postion of i
                vi = cfs[i]['statev'] # velocity of i
                # neighborhood trivially
                N_i = list(cfs.keys())
                N_i.remove(i)
                # for j in N_i.copy():
                #     pj = cfs[j]['statep'] # postion of j
                #     d_ij = np.linalg.norm(pi-pj)
                #     if d_ij > 1.6:
                #         N_i.remove(j)
                        
                # N_i = sorted(N_i, key = np.linalg.norm(pi-cfs[j]['statep']))        
                f_sep = np.zeros(3)
                f_coh = np.zeros(3)
                f_alig = np.zeros(3)
                f_obs = np.zeros(3)
                f_mig = np.zeros(3)

                for k, j in enumerate(sorted(N_i)):
                    pj = cfs[j]['statep'] # postion of j
                    vj = cfs[j]['statev'] # velocity of i
                    d_ij =np.linalg.norm(pi-pj)
                    if d_ij <= d_ref:
                        f_sep += W_sep*(d_ref-d_ij)*(pi-pj)/d_ij
                    else:
                        
                        f_coh += W_coh*(d_ij-d_ref)*(pj-pi)/d_ij

                    f_alig += W_alig*(vj-vi)*math.exp(-k/len(N_i))

                obstacles = [np.array([0.5,0.05,Z])]
                obstacles = []
                for m in obstacles:
                    pm = np.array(m)
                    dim = np.linalg.norm(pi-pm) # distance of i and m
                    vm = v_obs*(pi-pm)/dim
                    vim = np.linalg.norm(vi-vm) # relative speed
                    v_obsmax = D(dim-r_0, a_obs, p_obs)
                    if vim > v_obsmax:
                        f_obs += W_obs*(vim-v_obsmax)*(vm-vi)/dim
                # check dis
                # cfs[i]['dis'] = min(dim, cfs[i]['dis'])

                # global u_ref
                # if i == 5:
                #     if u_ref[0] == -1.0 and pi[0] < -1.20:
                #         u_ref = np.array([0, -1.0, 0])
                #     elif u_ref[1] == -1.0 and pi[1] < -1.20:
                #         u_ref = np.array([1.0, 0.0, 0])
                #     elif u_ref[0] == 1.0 and pi[0] > 1.20:
                #         u_ref = np.array([0, 1.0, 0])
                #     elif u_ref[1] == 1.0 and pi[1] > 1.20:
                #         u_ref = np.array([-1.0, 0.0, 0])       
                # f_mig = W_mig*(v_ref*u_ref-vi)
            
                
                if np.linalg.norm(pi-vertices[cfs[i]['goal_idx']]) < 0.3:
                    temp = list(range(len(vertices)))
                    temp.remove(cfs[i]['goal_idx'])
                    print(cfs[i]['goal_idx'])
                    cfs[i]['goal_idx'] = choice(temp)
                    print(cfs[i]['goal_idx'])
                
                    for j in N_i:
                        cfs[j]['goal_idx'] = cfs[i]['goal_idx']
                    
                cfs[i]['goal_dir'] = (vertices[cfs[i]['goal_idx']]-pi)/np.linalg.norm(vertices[cfs[i]['goal_idx']]-pi)
                f_mig = W_mig*(v_ref*cfs[i]['goal_dir']-vi)
                    
                # cycles = [np.array([0.0, 1.0, Z])]
                # if pi[0] < 0.05:
                #     cfs[i]['goal_dir'] = u_ref
                # else:
                #     cfs[i]['goal_dir'] = (cycles[0]-pi)/np.linalg.norm(cycles[0]-pi)
                # f_mig = W_mig*(v_ref*cfs[i]['goal_dir']-vi)
                
                # f_mig = np.zeros(3)
                # f_alig = np.zeros(3)
                a = (f_sep+f_coh+f_alig+ f_obs + f_mig)/mass

                a = a/np.linalg.norm(a)*min(np.linalg.norm(a), a_max)
                
                
                '''
                补偿下降后拉升不起来
                '''                
                # if vi[2] < 0 and a[2] > 0:
                #     a[2] = 1.5*a[2]
                
                '''
                update state based
                '''
                v = vi + a*dt
                v = v/np.linalg.norm(v)*min(np.linalg.norm(v), v_max)
                
                if max(abs(v)) > 0.4:
                    v = v*0.4/max(abs(v))
                    
                pos = pi + (v)/2*dt
                '''
                update setpoint based
                '''
                # v = cfs[i]['setpointv'] + a*dt
                # v = v/np.linalg.norm(v)*min(np.linalg.norm(v), v_max)
                # cfs[i]['setpointv'] = v
                # pos = cfs[i]['setpointp'] + (vi+v)/2*dt
                # cfs[i]['setpointp'] = pos  
                
                # if pi[0] < -1:
                #     # allcfs.crazyfliesById[i].cmdFullState(pos = np.array([-1.0, i-5 ,Z]), 
                #     # vel= np.zeros(3), acc = np.zeros(3), yaw=0.0, omega=np.zeros(3))
                #     allcfs.crazyfliesById[i].cmdFullState(pos = pi, 
                #     vel= np.zeros(3), acc = np.zeros(3), yaw=0.0, omega=np.zeros(3))
                # else:
                    
                if 1:
                # command 
                    allcfs.crazyfliesById[i].cmdFullState(pos = pos, 
                        vel= v, acc = a, yaw=0.0, omega=np.zeros(3))           
                    with open(file=file_name, mode='a+') as f:
                        f.write(f'{i} pi{pi} a{a} vi{vi} {v}\n')
                        # f.write(f'{i} pi{pi[:2]} {pos[:2]} a{a[:2]} vi{vi[:2]} {v[:2]}\n')
                # if pi[0] > 1:
                #     pos = np.array(pi)
                #     pos[0] = pi[0]-self.count*dt {pos[:2]} 
                #     allcfs.crazyfliesById[i].cmdFullState(pos = pos, 
                #     vel= np.array([-1.0,0,0]), acc = np.array([-0.1,0,0]), yaw=0.0, omega=np.zeros(3))
                #     # with open(file=file_name, mode='a+') as f:
                #     #     f.write(f'{self.count} {i} {pi[0]-pos[0]} {pi} {pos}\n')
                # elif pi[0] < -1:
                #     allcfs.crazyfliesById[i].cmdFullState(pos = np.array([-1.0, i-5 ,Z]), 
                #     vel= np.zeros(3), acc = np.zeros(3), yaw=0.0, omega=np.zeros(3))
                # else:
                # # command 
                #     allcfs.crazyfliesById[i].cmdFullState(pos = pi+dt*v, 
                #         vel= v, acc = a, yaw=0.0, omega=np.zeros(3))
                
                # allcfs.crazyfliesById[i].cmdPosition(pos=pi+dt*v)
                

                    # f.write(f'{self.count} {i} {pi[0]-pos[0]:.3f} {pi} {pos} a{a} v{v}import numpy as np{vi}\n')
                # if pi[0] > 0 and pi[0] < 2:
                #     pos = np.array(pi)
                #     if i % 2 != 0:
                #         pos[0] = 1-self.count*dt
                #     else:
                #         pos[0] = 1+self.count*dt
                #     allcfs.crazyfliesById[i].cmdPosition(pos=pos)
                #     with open(file='logp4', mode='a+') as f:
                #         f.write(f'{self.count} id{i} now{pi} {pos}\n')
                # # elif pi[0] > -1:
                # #     allcfs.crazyfliesById[i].cmdPosition(pos=pi+v*dt)
                # else:
                #     allcfs.crazyfliesById[i].cmdPosition(pos=pi)
                #     with open(file='logp5', mode='a+') as f:
                #         f.write(f'{self.count} id{i} now{pi}\n')
                
                self.count += 1

    node = TimeSyncNode()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    
    s = input()
    if s == '':
        allcfs.emergency()
        print("XXXXXXXXXXXXXXXXXX")
        
    print(node.dis)    
    node.destroy_node()
    
    
    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.08, duration=1.0 + Z)
    timeHelper.sleep(2.0 + Z)

    print(cfs)
if __name__ == "__main__":
    main()

