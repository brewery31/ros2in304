# ros2in304

## TDFC 圆环与螺旋线实验

独立 [`tdfc_lab`](tdfc_lab/README.md) 包当前只保留四机/六机圆环、四机/六机螺旋线，均使用我们的局部曲线控制器。字母实验已停止，历史失败数据保留归档。先完成固件仿真验收，再进行真机接入。

Crazyflie（crazyswarm2）无人机集群实验代码快照 —— 机房工作机 `/home/crazyswarm2/ros2_ws` 的工作现场导出，用于在其他机器上查看/复现。

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/IMRCLab/crazyswarm2 |
| 上游 commit | `1069b67` (2024-10-18, Merge pull request #578 from IMRCLab/arm_close_fix) |
| 子模块 | `crazyflie/deps/crazyflie_tools` @ `a2d74b4`（已以普通文件形式包含） |
| ROS 发行版 | humble |

本仓库 = 上述 commit 的完整源码树（去掉 `.git`）**＋** 本机未提交的改动。因此：
**不要**把它当成可以 `git pull` 跟随上游的分支，升级上游请直接重新 clone。

## 相对上游的改动

已修改（`M`）：

- `crazyflie/config/crazyflies.yaml` — 机队使能、每架 cf 的 radio URI(`radio://0/80/2M/E7E7E7E7xx`)、初始位置
- `crazyflie/config/motion_capture.yaml` — 动捕类型 `qualisys`、主机 `192.168.254.102`、marker 配置
- `crazyflie_examples/setup.cfg` — 新增 console_scripts：`lf` `test` `figure` `sor` `test2` `go`
- `crazyflie_examples/crazyflie_examples/hello_world.py`
- `crazyflie_examples/crazyflie_examples/nice_hover.py`
- `crazyflie_examples/crazyflie_examples/cmd_full_state.py`
- `crazyflie_examples/crazyflie_examples/data/multi_trajectory/traj0.csv`

新增（`??`），都在 `crazyflie_examples/crazyflie_examples/`：

| 文件 | 用途 |
|---|---|
| `SOR.py` | 集群编队/避碰实验（`sor` 入口） |
| `sor_formation.py` | 编队控制（`go` 入口） |
| `simtoreal_10uavs.py` | 10 机 sim-to-real 轨迹回放 |
| `lf.py` | 领导者-跟随者（leader-follower）实验（`lf` 入口） |
| `test2.py` | 单机起飞-悬停-降落 + 位姿订阅 |
| `test.py` / `1.py` | 早期测试脚本 |
| `figure` | 轨迹绘图脚本（`figure` 入口） |
| `cmd_full_state_copy.py` | `cmd_full_state.py` 的副本 |
| `data/figure.csv`、`data/multi_trajectory/traj0 copy.csv` | 轨迹数据 |
| `data/simtoreal_10uavs/` | 10 机仿真结果 json（`final_positions_*.json`） |

`ros2_ws_extra/` 是工作空间根目录下另存的文件：

- `tmp_server.yaml`、`tmp_motion_capture.yaml` — `crazyflie_server` / 动捕的调参副本
- `mc_help.py` — 直连动捕看刚体位置的调试脚本
- `params*.csv` — Crazyflie 参数导出（activeMarker 等）

## 未包含

- `build/ install/ log/ cache/`、`log2`：colcon 构建产物与日志，机器相关，请自行 `colcon build`
- `src/crazyswarm2/crazyflie_examples/.../data/simtoreal_10uavs.zip`（9.2 MB）：与上面的 json 目录重复

## 编译

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/brewery31/ros2in304.git    # 或直接拷贝本目录内容
# 还需要另外 clone 依赖：BehaviorTree.CPP、BehaviorTree.ROS2、generate_parameter_library、tl_expected
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash
```

## 运行（真机）

```bash
ros2 launch crazyflie launch.py backend:=cflib \
  crazyflies_yaml_file:=<path>/crazyflies.yaml \
  motion_capture_yaml_file:=<path>/motion_capture.yaml
ros2 run crazyflie_examples sor     # 或 go / lf / test2 / figure
```
