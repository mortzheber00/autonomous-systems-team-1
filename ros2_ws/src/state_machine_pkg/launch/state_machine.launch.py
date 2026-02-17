from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("state_machine_pkg"),
        "config",
        "state_machine.yaml",
    ])

    state_machine_node = Node(
        package="state_machine_pkg",
        executable="state_machine_node",
        name="state_machine_node",
        output="screen",
        parameters=[config_file, {"use_sim_time": False}],
    )

    waypoint_planner_node = Node(
        package="basic_waypoint_pkg",
        executable="basic_waypoint_node",
        name="planner",
        output="screen",
        parameters=[config_file, {"use_sim_time": False}],
    )

    sampler_node = Node(
        package="mav_trajectory_generation",
        executable="trajectory_sampler_node",
        name="sampler",
        output="screen",
        parameters=[{"use_sim_time": False}],
        remappings=[
            ("path_segments_4D", "trajectory"),  # Planner publishes to "trajectory"
        ],
    )

    local_planner_node = Node(
        package="planner_pkg",
        executable="local_planner_node",
        name="local_planner_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    return LaunchDescription([
        state_machine_node,
        local_planner_node,
        waypoint_planner_node,
        sampler_node,
    ])
