"""Only in-process simulation; no hardware backend selectable here."""
from datetime import datetime
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package = Path(get_package_share_directory('tdfc_lab'))
    return LaunchDescription([
        DeclareLaunchArgument('shape',default_value='ring',choices=['ring','helix','R','A','L']),
        DeclareLaunchArgument('agents',default_value='4',choices=['4','5','6']),
        DeclareLaunchArgument('seed',default_value='0'),
        DeclareLaunchArgument('formation_seconds',default_value='60.0'),
        DeclareLaunchArgument('rviz',default_value='True'),
        DeclareLaunchArgument('output_dir',default_value=str(Path.home()/'tdfc_sim_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))),
        Node(package='tdfc_lab',executable='tdfc_sim',output='screen',parameters=[{
            'shape':ParameterValue(LaunchConfiguration('shape'),value_type=str),
            'agents':ParameterValue(LaunchConfiguration('agents'),value_type=int),
            'seed':ParameterValue(LaunchConfiguration('seed'),value_type=int),
            'formation_seconds':ParameterValue(LaunchConfiguration('formation_seconds'),value_type=float),
            'output_dir':ParameterValue(LaunchConfiguration('output_dir'),value_type=str),
        }]),
        Node(package='rviz2',executable='rviz2',condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d',str(package/'config/sim.rviz')],output='screen'),
    ])
