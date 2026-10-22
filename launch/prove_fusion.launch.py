#!/usr/bin/env python3
"""
prove_fusion.launch.py
----------------------
Starts the full stack on the Jetson:
  - detector_node    (YOLO on 3 cameras)
  - fusion_node      (camera-LiDAR frustum fusion)
  - lane_node        (classical lane detection on front camera)
  - viz_node         (3D box markers)
  - rviz2            (preloaded with prove_fusion.rviz)

All nodes read /path/to/config/sensors.yaml so you edit calibration in
exactly one place.

Launch args:
  rviz:=true|false       - start RViz on this machine (default true)
  config:=<path>         - override sensors.yaml location
  frame_skip:=<int>      - process every Nth camera frame (default 1)

Typical run on the Jetson:
  ros2 launch prove_fusion prove_fusion.launch.py

Typical run on the Jetson, display RViz on a laptop over the net instead:
  on Jetson:  ros2 launch prove_fusion prove_fusion.launch.py rviz:=false
  on laptop:  rviz2 -d $(ros2 pkg prefix prove_fusion)/share/prove_fusion/rviz/prove_fusion.rviz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('prove_fusion')

    default_config = PathJoinSubstitution([pkg, 'config', 'sensors.yaml'])
    default_rviz = PathJoinSubstitution([pkg, 'rviz', 'prove_fusion.rviz'])

    arg_rviz = DeclareLaunchArgument('rviz', default_value='true')
    arg_config = DeclareLaunchArgument('config', default_value=default_config)
    arg_frame_skip = DeclareLaunchArgument('frame_skip', default_value='1')

    config = LaunchConfiguration('config')

    detector = Node(
        package='prove_fusion',
        executable='detector_node',
        name='prove_detector',
        output='screen',
        emulate_tty=True,
        parameters=[config, {'frame_skip': LaunchConfiguration('frame_skip')}],
    )

    fusion = Node(
        package='prove_fusion',
        executable='fusion_node',
        name='prove_fusion',
        output='screen',
        emulate_tty=True,
        parameters=[config],
    )

    lane = Node(
        package='prove_fusion',
        executable='lane_node',
        name='prove_lane',
        output='screen',
        emulate_tty=True,
        parameters=[config],
    )

    viz = Node(
        package='prove_fusion',
        executable='viz_node',
        name='prove_viz',
        output='screen',
        emulate_tty=True,
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', default_rviz],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        arg_rviz, arg_config, arg_frame_skip,
        GroupAction([detector, fusion, lane, viz, rviz]),
    ])
