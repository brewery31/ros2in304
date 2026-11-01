import numpy as np
from crazyflie_py import Crazyswarm
import rclpy
import message_filters
import threading
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from itertools import combinations

# 配置参数
TAKEOFF_DURATION = 2.0
FLIGHT_HEIGHT = 0.3
CONTROL_FREQUENCY = 60.0  # Hz
DT = 1.0 / CONTROL_FREQUENCY

# 目标点参数
TARGET_POINT = np.array([-1.0, 0.0, FLIGHT_HEIGHT])  # 目标点坐标
TARGET_ATTRACTION_RADIUS = 0.2  # 吸引力的作用半径
TARGET_REPULSION_RADIUS = 0.2 # 排斥力的作用半径
TARGET_FORCE_WEIGHT = 1.0  # 目标力的权重

# Boid算法参数
class BoidParams:
    def __init__(self):
        self.min_speed = 0.2
        self.max_speed = 1.0
        self.perception_radius = 1.5
        self.avoidance_radius = 1
        self.max_steer_force = 1.5

        self.align_weight = 1.0
        self.cohesion_weight = 1.0
        self.separate_weight = 1.5

        self.formation_force_weight = 2.0
        self.center_force_weight = 2.0
        self.collision_avoid_weight = 5.0
        self.target_force_weight = TARGET_FORCE_WEIGHT  # 添加目标力权重

        self.center_desired_distance = 0.5
        self.center_switch_threshold = 1.0


def main():
    swarm = Crazyswarm()
    time_helper = swarm.timeHelper
    all_cfs = swarm.allcfs
    cf_list = all_cfs.crazyflies

    # 初始化无人机状态
    cfs = {}
    for i, cf in enumerate(cf_list):
        cf_id = int(cf.uri[-2:], 16)
        # 根据索引分配不同的formation_offset
        if i == 0:
            formation_offset = np.array([0, 1, 0])  # 第一个无人机
        else:
            formation_offset = np.array([0, -1, 0])   # 第二个无人机
        
        cfs[cf_id] = {
            'position': None,
            'velocity': None,
            'forward': np.array([1.0, 0, 0]),
            'formation_offset': formation_offset,
            'team_id': 0  # 所有无人机同队
        }

    # 起飞
    all_cfs.takeoff(targetHeight=FLIGHT_HEIGHT, duration=TAKEOFF_DURATION)
    time_helper.sleep(TAKEOFF_DURATION + 1.0)

    class BoidControllerNode(rclpy.node.Node):
        def __init__(self):
            super().__init__("boid_controller")
            self.params = BoidParams()
            self.target_point = TARGET_POINT  # 存储目标点

            # 创建消息同步器
            self.subscribers = []
            for cf_id in cfs.keys():
                self.subscribers.append(
                    message_filters.Subscriber(self, Odometry, f"/cf{cf_id}/odom")
                )

            self.time_sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, 10, 1.0 / CONTROL_FREQUENCY
            )
            self.time_sync.registerCallback(self.control_callback)

            self.collision_log = []

        def get_neighbors(self, cf_id, current_pos):
            """获取当前无人机的邻居"""
            neighbors = []
            for other_id, other_data in cfs.items():
                if other_id == cf_id:
                    continue

                distance = np.linalg.norm(current_pos - other_data['position'])
                if distance < self.params.perception_radius and distance > 0:
                    neighbors.append(other_id)
            return neighbors

        def calculate_target_force(self, position):
            """计算目标点的吸引力/排斥力"""
            to_target = self.target_point - position
            distance = np.linalg.norm(to_target)
            
            if distance < TARGET_REPULSION_RADIUS:
                # 距离太近，产生排斥力
                force = (TARGET_REPULSION_RADIUS - distance) * (-to_target / (distance + 1e-6))
            elif distance > TARGET_ATTRACTION_RADIUS:
                # 距离太远，产生吸引力
                force = (distance - TARGET_ATTRACTION_RADIUS) * (to_target / (distance + 1e-6))
            else:
                # 在理想范围内，不产生力
                force = np.zeros(3)
                
            return force * self.params.target_force_weight

        def calculate_forces(self, cf_id, neighbors):
            """计算Boid算法的各种力"""
            current_data = cfs[cf_id]
            position = current_data['position']
            velocity = current_data['velocity']

            alignment_force = np.zeros(3)
            cohesion_force = np.zeros(3)
            separation_force = np.zeros(3)
            formation_force = np.zeros(3)
            collision_avoid_force = np.zeros(3)
            target_force = self.calculate_target_force(position)  # 计算目标力

            center_of_mass = np.zeros(3)
            num_neighbors = len(neighbors)

            if num_neighbors > 0:
                # 计算邻居的中心位置
                for neighbor_id in neighbors:
                    neighbor_pos = cfs[neighbor_id]['position']
                    neighbor_vel = cfs[neighbor_id]['velocity']
                    center_of_mass += neighbor_pos

                    # 对齐力
                    alignment_force += neighbor_vel

                    # 分离力和内聚力
                    distance = np.linalg.norm(position - neighbor_pos)
                    if distance < self.params.avoidance_radius:
                        separation_force += ((self.params.avoidance_radius - distance) / distance) * \
                                            (position - neighbor_pos)
                    else:
                        cohesion_force += neighbor_pos - position

                center_of_mass /= num_neighbors
                alignment_force /= num_neighbors

                # 编队力
                desired_position = center_of_mass + current_data['formation_offset']
                offset = desired_position - position
                formation_force = offset * self.params.formation_force_weight

                # 简单碰撞检测
                if self.check_collision(position, velocity):
                    collision_avoid_force = self.avoid_collision(position)

            # 计算合力
            total_force = (
                    self.params.align_weight * alignment_force +
                    self.params.cohesion_weight * cohesion_force +
                    self.params.separate_weight * separation_force +
                    formation_force +
                    self.params.collision_avoid_weight * collision_avoid_force +
                    target_force  # 添加目标力
            )

            # 限制力的大小
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > self.params.max_steer_force:
                total_force = total_force / force_magnitude * self.params.max_steer_force

            return total_force

        def check_collision(self, position, forward):
            """简单碰撞检测"""
            # 这里可以添加更复杂的碰撞检测逻辑
            return False

        def avoid_collision(self, position):
            """避障行为"""
            # 这里可以添加更复杂的避障逻辑
            return np.random.uniform(-1, 1, size=3)

        def control_callback(self, *odom_msgs):
            """控制回调函数"""
            # 更新无人机状态
            for i, cf_id in enumerate(cfs.keys()):
                position = np.array([
                    odom_msgs[i].pose.pose.position.x,
                    odom_msgs[i].pose.pose.position.y,
                    odom_msgs[i].pose.pose.position.z
                ])

                velocity = np.array([
                    odom_msgs[i].twist.twist.linear.x,
                    odom_msgs[i].twist.twist.linear.y,
                    odom_msgs[i].twist.twist.linear.z
                ])

                cfs[cf_id]['position'] = position
                cfs[cf_id]['velocity'] = velocity
                if np.linalg.norm(velocity) > 0:
                    cfs[cf_id]['forward'] = velocity / np.linalg.norm(velocity)

            # 为每架无人机计算控制指令
            for cf_id in cfs.keys():
                position = cfs[cf_id]['position']
                velocity = cfs[cf_id]['velocity']

                # 获取邻居
                neighbors = self.get_neighbors(cf_id, position)

                # 计算Boid力
                total_force = self.calculate_forces(cf_id, neighbors)

                # 计算加速度、速度和位置
                acceleration = total_force 
                new_velocity = velocity + acceleration * DT

                # 限制速度
                speed = np.linalg.norm(new_velocity)
                if speed > self.params.max_speed:
                    new_velocity = new_velocity / speed * self.params.max_speed
                elif speed < self.params.min_speed:
                    new_velocity = new_velocity / (speed + 1e-6) * self.params.min_speed
                acceleration = (new_velocity - velocity) / DT
                new_position = position + new_velocity * DT

                # 发送控制指令
                all_cfs.crazyfliesById[cf_id].cmdFullState(
                    pos=new_position,
                    vel=new_velocity,
                    acc=acceleration,
                    yaw=0.0,
                    omega=np.zeros(3)
                )

                # 记录碰撞事件
                for other_id in neighbors:
                    distance = np.linalg.norm(position - cfs[other_id]['position'])
                    if distance < 0.3:  # 碰撞阈值
                        self.collision_log.append(
                            f"Collision between {cf_id} and {other_id} at distance {distance:.2f}m")

    # 启动控制节点
    node = BoidControllerNode()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    # 等待用户输入停止
    input("Press Enter to stop...")
    node.destroy_node()

    # 降落
    for cf in all_cfs.crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.08, duration=2.0)
    time_helper.sleep(2.0)

    # 打印碰撞日志
    if node.collision_log:
        print("\nCollision events:")
        for log in node.collision_log:
            print(log)


if __name__ == "__main__":
    main()