from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


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

    global_planner_node = Node(
        package="planner_pkg",
        executable="global_planner_node",
        name="global_planner_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    rrt_node = Node(
        package="multidim_rrt_cpp",
        executable="rrt3D",
        name="rrt3D",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    controller_node = Node(
        package="controller_pkg",
        executable="controller_node",
        name="controller_node",
        output="screen",
        parameters=[{'use_sim_time': True}],
    )

    lantern_location_node = Node(
        package="lantern_location",
        executable="lantern_location_node",
        name="lantern_location_node",
        output="screen",
        parameters=[{'use_sim_time': True}],
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("occupancy_grid"), '/launch', '/mapping.launch.py'
        ]),
    )


    return LaunchDescription([
        controller_node,
        lantern_location_node,
        mapping_launch,
        state_machine_node,
        local_planner_node,
        global_planner_node,
        rrt_node,
        waypoint_planner_node,
        sampler_node,
    ])
