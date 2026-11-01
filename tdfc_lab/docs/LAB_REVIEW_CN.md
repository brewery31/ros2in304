# 机房 ROS 2 仓库检查与接入建议

检查日期：2026-11-01。仓库：https://github.com/brewery31/ros2in304

检查快照：`3b786c4d79acba616940b0bfcfe86ca8d6d65d72`。
本地只读检查副本：`/Users/brewery/workspace/ros2in304`。未启动 ROS、无线电、仿真或真机命令，未修改源文件。检查属于静态审阅，不能代替机房运行验证。

## 已确定的配置

- README 指定 ROS 2 Humble，来源是 Crazyswarm2 上游 1069b67 的机房快照。
- 主配置 robot type 为 cf21、big_quad=false；可确定 Crazyflie 平台，具体硬件修订、载荷和固件版本仍需机房读取。
- Qualisys：192.168.254.102，单 marker 配置；动捕 QoS deadline 为 100 Hz，是配置目标，不是实测速率。
- 主 crazyflies.yaml 启用 cf5、cf6、cf7、cf8、cf9、cfA、cfB、cfC、cfD、cfE，共 10 架，使用 radio0/channel80 与 radio1/channel90。
- 初始位置为 1 m 间隔的地面网格，范围 x∈[-1,2]、y∈[-1,1]。这是配置摆位，不能据此推断整个机房可用飞行边界。
- 主配置：Kalman estimator=2、Mellinger controller=2、odom 日志目标 60 Hz，HighLevel 启用。

## 实际链路

Qualisys → motion_capture_tracking 的 /poses → crazyflie_server → 无线外部定位 → 机载估计器。

机载状态日志 → /<配置中的机器人名>/odom → 主机控制程序 → /<机器人名>/cmd_full_state → cflib send_full_state_setpoint → 机载控制器。

Python API 已有 takeoff、goTo、cmdFullState、notifySetpointsStop、land、emergency。emergency 是停电机，并非正常受控降落。

## 接入前必须处理的问题

1. **机器人命名不一致。** SOR.py、sor_formation.py、test2.py 用 radio URI 末两位转十进制再构造 /cf{id}/odom。cfA 将变成 /cf10/odom，而 server 按 YAML 名称发布 /cfA/odom；cfB–cfE 同样不匹配。全机消息同步因此可能一直等不到回调。应使用 crazyfliesByName 和原始 YAML 名称建立显式映射。
2. **速度接口不能直接调用。** Python cmdVelocityWorld 及其 publisher 被注释，cflib server 当前也没有相应世界速度订阅。可采用现有 cmdFullState，但必须明确位置、速度、加速度参考的生成方式，并重新验证等效闭环；不能把纯速度仿真的保证原样搬过去。
3. **odom 时间戳是接收/发布时刻。** _log_odom_data_callback 收到固件 timestamp 却用主机 now() 填消息 header。这不表示状态测量年龄为零。应记录原始固件时间、接收时间及其时钟映射；仅看 ROS header 不能标定端到端延迟。
4. **状态语义需实测。** odom 速度来自 kalman.statePX/PY/PZ。需核对当前固件下单位和坐标系，并用动捕位置差分交叉校验。消息 frame_id 不能单独证明数值坐标语义。
5. **配置副本相互冲突。** tmp_server.yaml 是两机 cf4/cf5、PID controller=1、pose 10 Hz；主配置是十机、Mellinger、odom 60 Hz。tmp_motion_capture 的 marker 偏置也不同。必须固定一套显式配置，不能混用临时副本。
6. **simtoreal_10uavs.py 未完成。** json.load() 缺少文件参数，且 main 在创建 swarm 前直接 return；文件名不能作为已完成十机实飞的证据。
7. **旧脚本不等于当前控制器。** SOR/go 是旧规则，并有固定 dt、全机近似同步依赖。未见这些脚本具备完整的新鲜度检查、失联监管和异常清理状态机。test2 的 SHOULD_LAND 分支使用未创建的 notify_client/land_client。不要直接拿这些作为当前三场景的部署入口。
8. **起飞与降落仍需独立设计。** 已有 API 可复用，但现有脚本主要固定等待后切入控制；应改为反馈确认悬停。停止低层流后按 notifySetpointsStop 的有效期和服务完成情况交接高层，返航到分离降落柱后再下降，不能从螺旋终态直接假定垂直降落安全。

## 对当前研究的意义

可以复用机房的底层飞行与定位设施，无需重新选平台。主机上按机器人独立计算动作，与算法的局部计算结构兼容；但所有机器人状态由动捕集中提供并通过 ROS/无线收发，这种实飞不能写成“无外部基础设施通信”或“机载自主感知”。准确描述应是“基于外部定位的主机部署，按局部邻域独立计算，算法不需要机器人之间交换控制意图”；局部邻域必须由适配器实际限制。

已有离线 ego 接口的 20 ms 决策周期，与机房 60 Hz 日志并非一一对齐；需固定 50 Hz timer 与异步状态缓存，显式处理信息年龄，不能把全机回调间隔硬当 20 ms。5 ms 本机速度年龄门槛属于离线占位值，60 Hz 回传下不能未经标定直接沿用。

## 建议的实施顺序

1. 新建四机专用配置，只启用明确指定的四架，绑定 YAML 名称、机体与地面标记；保留原配置。
2. 仅采集不发送动作：记录 odom/动捕时间戳、速率、丢包、状态噪声，核对坐标、速度单位、marker 偏置与房间边界。
3. 用录包驱动当前 ego 接口做离线适配回放，检查动作限幅、信息年龄和控制开销。
4. 单机确认起飞—反馈悬停—新接口接管—停止低层流—降落，并测量实际速度环响应。
5. 将接口与标定后的动力学用于四机完整任务仿真，再进行四机水平圆环；其后倾斜圆环、螺旋线。六/八机作为后续扩展。

## 本次验证范围

- 四个实验脚本 AST 语法检查通过；未 import 或执行其 main。
- 按主配置逐架核对 URI→脚本话题：前五架匹配，cfA–cfE 共五架不匹配。
- 本地 Git 工作区干净。未编译 ROS 包、未验证 firmware 日志变量、未运行飞行、未改动已有封存实验。
