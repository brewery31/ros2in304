# TDFC 机房实验：先仿真，后真机

当前分支在 `581c418`（机房 dsh 改动）基础上修正。主实验是：

| 形状 | 飞机数量 | 方法 |
|---|---|---|
| 水平圆环 ring | 4、6 | 连续曲线控制器 curve_ours |
| 单圈空间螺旋线 helix | 4、6 | 连续曲线控制器 curve_ours |
| 字母 R、A、L | 每个字母 5 架，分别运行 | 与圆环/螺旋线相同的局部曲线/笔画控制器 |

字母实验现已移除集中式匹配和逐机固定目标。R、A、L 使用完整的公共笔画图作为形状先验，每架机器人调用同一 `CurveEgoInterface → curve_acceleration → ORCA → governor`，根据自身及可见邻居状态贴合、铺展。没有 Hungarian 分配，也不指定哪架飞机占哪个角。

这是我们算法的**曲线/笔画适配版**，不是曲面原算法原封不动搬到分叉字母上。字母交点和尖角处切向可能不唯一，原光滑曲线结论不能直接推广。笔画总长度与圆环/螺旋线采用同一 `0.7N` 规则；不按失败结果调小安全距离或预先分配站位。图中灰线是公共形状，彩色点才是实际机器人。

上一版的五关键点字母方案不是本文算法，已撤回相应九次成功及混合“21/21”结论；原结果仅保留为带警示的历史记录。当前结果以 `results/local_strokes_20261102/` 和 `docs/LOCAL_LETTERS_20261102_CN.md` 为准，失败照实保留。

当前统一算法的21次数值回归：圆环/螺旋线12/12成形，五机字母0/9成形；全部没有间距低于0.5m。24项测试通过代表实现和回归检查通过，不代表字母性能通过。字母还需在同一控制框架内改进。

## 机房：运行固件/刚体仿真

只运行下述 `tdfc_lab sim.launch.py`，不需另开原来的 crazyflie_server、动捕或真机脚本。

先在仓库目录更新（本机改动需先自行提交保留，不要 reset）：

```bash
git switch codex/tdfc-lab-integration
git pull --ff-only
```

在 ROS 工作区根目录编译。下例假定使用独立的 `~/tdfc_sim_ws`；如果仓库在原工作区，就改为对应的工作区路径。

```bash
source /opt/ros/humble/setup.bash
cd ~/tdfc_sim_ws
colcon build --symlink-install --packages-up-to tdfc_lab \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
python3 -c "import cffirmware, rowan, numpy, scipy; print('SIL dependencies OK')"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=73
```

依次单独运行，每次实验结束后 Ctrl+C 退出再换场景：

```bash
ros2 launch tdfc_lab sim.launch.py shape:=ring agents:=4
ros2 launch tdfc_lab sim.launch.py shape:=ring agents:=6
ros2 launch tdfc_lab sim.launch.py shape:=helix agents:=4
ros2 launch tdfc_lab sim.launch.py shape:=helix agents:=6
ros2 launch tdfc_lab sim.launch.py shape:=R agents:=5
ros2 launch tdfc_lab sim.launch.py shape:=A agents:=5
ros2 launch tdfc_lab sim.launch.py shape:=L agents:=5
```

上述是七条备选命令，不要同时启动七个场景。默认自动打开 RViz；无显示器可加 `rviz:=False`。可加 `seed:=1` 或 `formation_seconds:=30.0`。

每次自动建立 `~/tdfc_sim_runs/<时间戳>/`，含 `trajectory.csv`、`summary.json`，不能覆盖同名目录。阶段为：同高网格地面摆位 → 10 秒垂直起飞参考 → 根据状态确认悬停 → 成形 60 秒 → 返回各自分离起降柱 → 10 秒降落参考。起飞/返回/降落超时或检测到碰撞会终止本次仿真并保留失败结果，不伪造成功。

**实现方式**：`tdfc_sim` 在同一进程直接调用本仓库的 `CrazyflieSIL`（cffirmware Mellinger）与 `crazyflie_sim.backend.np`（0.5 ms 刚体积分），每 20 ms 更新一次任务控制。不连接 radio，不读取飞机 URI，不调用真机 ROS 服务；发布 `/tdfc_sim/<name>/odom` 和 `/tdfc_sim/markers`。因此无需给旧仿真服务器补 odom 或用重映射将候选指令转发给飞机。

默认观察是精确模拟状态，没有自动加入动捕噪声/延迟。速度设定值通过带 0.27m/s 参考速度预留的预测器转换为位置、速度、加速度参考；实际刚体速度另行记录，不能把参考限幅当实际约束证明。`summary.json` 中的最小距离是每 0.5 ms 的采样最小值，不是连续时间解析证书。

**验证边界**：开发机没有 ROS、cffirmware、rowan。本地完成的是数值模型回归、任务逻辑与接口测试；这个固件/刚体入口仍需在机房运行验收。不要将未运行的 SIL 写成已通过，更不能据此起飞真机。

## 任意电脑：先跑轻量数值仿真

需要 C++ 编译器、NumPy、SciPy；画图另需 Matplotlib。在仓库根目录：

```bash
sh tdfc_lab/scripts/build_offline_orca.sh
PYTHONPATH=tdfc_lab python3 -m unittest discover -s tdfc_lab/test -v
PYTHONPATH=tdfc_lab python3 -m tdfc_lab.offline --shape ring --agents 6 --steps 3000 --plot /tmp/ring6.png
PYTHONPATH=tdfc_lab python3 -m tdfc_lab.offline --shape helix --agents 4 --steps 3000 --plot /tmp/helix4.png
PYTHONPATH=tdfc_lab python3 -m tdfc_lab.offline --shape R --agents 5 --steps 3000 --plot /tmp/R5.png
```

编译过 ROS 包后也可用 `ros2 run tdfc_lab tdfc_offline` 替代 `PYTHONPATH=... python3 -m tdfc_lab.offline`。

全部七个场景，各三个固定种子：

```bash
ros2 run tdfc_lab tdfc_offline --suite --steps 3000 --out /tmp/tdfc_scenes_new
```

输出目录必须不存在。所有运行都保留；固定 seed0 画图，不按结果挑最好案例。`--csv` 导出每步位置，`--trace` 导出 pos/vel/time NPZ。轻量模型从初始悬停开始，不包括真实起飞与刚体姿态控制。

当前统一算法数值结果见 `results/local_strokes_20261102/`，判据见其 plan.json。原归档迁移回归仍使用三份 200 步观测夹具；跨平台 native ORCA 命令容差为 1e-7 m/s，并拒绝 NaN 期望值和真正的大偏差。

## 修正与出处

- 六机/五机不再只在底层“名义支持”，入口、ID、初始摆位、作图、回放均按 N 处理。
- 阻尼/低通只作用于 ORCA 与 governor 之前的期望动作；最终 governor 输出不再被二次改写。默认仍关闭这两个实验开关，不凭单例自动采用新调参。
- 字母恢复为与圆环/螺旋线同一控制器，目标是公共笔画图；不会把抬笔处连成额外斜线。
- 字母成形没有集中优化、站位匹配或逐机目标。返航只返回各机自己的原始起降柱，也没有目标匹配。模拟器统一调度不代表控制律采用集中式目标分配。
- 原四机 tasks.json 和旧数值夹具保留；六机场景由相同参数和明确的 0.7N 曲线长度规则生成。
- SOURCE_MANIFEST.json 保存来源和修改说明；docs 中旧日期文档是历史依据，当前使用以本文为准。

## 后续真机

观察节点 `tdfc_observe` 仍只发候选话题，不会发送飞机动作；其 ROS odom header 仍是接收时间，尚未完成真实测量时间标定。四/五/六机 YAML 草案仅供核对，SIL 不使用这些 URI。

必须先在机房验证固件仿真，之后确认机体/动捕身份、房间边界、下洗间距、坐标、时钟和速度环，再做单机接口接管与真实起降状态机。当前分支没有开放 real 模式，实飞仍为 0。不能把 SIM 的速度环模型或精确状态保证当成实机安全证书。
