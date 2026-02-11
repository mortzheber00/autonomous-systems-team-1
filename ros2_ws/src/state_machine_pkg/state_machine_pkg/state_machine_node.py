#!/usr/bin/env python3

import json
import math
import os
from enum import Enum
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.exceptions import ParameterUninitializedException

from geometry_msgs.msg import Quaternion, Transform, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

from ament_index_python.packages import get_package_share_directory


class MissionState(Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    GO_TO_ENTRANCE = "go_to_entrance"
    EXPLORE = "explore"
    LAND = "land"
    DONE = "done"


class StateMachineNode(Node):
    def __init__(self):
        super().__init__("state_machine_node")

        self.declare_parameter("auto_start", True)
        self.declare_parameter("takeoff_alt", 10.0)
        self.declare_parameter("entry_goal", [-55.0, 10.0, 12.0])
        self.declare_parameter("land_goal", [])
        self.declare_parameter("land_z", 1.0)
        self.declare_parameter("goal_tolerance", 2.0)
        self.declare_parameter("command_rate_hz", 2.0)
        self.declare_parameter("target_lantern_count", 3)
        self.declare_parameter("explore_timeout_sec", 600.0)
        self.declare_parameter("lantern_output_path", "")
        self.declare_parameter("use_basic_waypoint_entry", False)

        self.auto_start = self.get_parameter("auto_start").get_parameter_value().bool_value
        self.takeoff_alt = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        self.entry_goal = self._get_vec_param("entry_goal")
        self.land_goal = self._get_vec_param("land_goal", optional=True)
        self.land_z = self.get_parameter("land_z").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.command_rate_hz = self.get_parameter("command_rate_hz").get_parameter_value().double_value
        self.target_lantern_count = self.get_parameter("target_lantern_count").get_parameter_value().integer_value
        self.explore_timeout_sec = self.get_parameter("explore_timeout_sec").get_parameter_value().double_value
        self.lantern_output_path = self._resolve_output_path(
            self.get_parameter("lantern_output_path").get_parameter_value().string_value
        )
        self.use_basic_waypoint_entry = (
            self.get_parameter("use_basic_waypoint_entry").get_parameter_value().bool_value
        )

        self.state = MissionState.IDLE
        self.have_odom = False
        self.current_pose = None
        self.current_orientation = None
        self.current_yaw = 0.0

        self.explore_start_time = None
        self.land_target = None

        self.detected_lanterns: Dict[int, Dict[str, float]] = {}

        self.trajectory_pub = self.create_publisher(
            MultiDOFJointTrajectory,
            "/command/trajectory",
            10,
        )

        enable_qos = QoSProfile(depth=1)
        enable_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.enable_local_planner_pub = self.create_publisher(
            Bool,
            "/state_machine/enable_local_planner",
            enable_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/current_state_est",
            self.odom_callback,
            10,
        )

        self.lantern_sub = self.create_subscription(
            MarkerArray,
            "/lantern_location/detected_lanterns",
            self.lantern_callback,
            10,
        )

        timer_period = 1.0 / max(self.command_rate_hz, 0.1)
        self.timer = self.create_timer(timer_period, self.on_timer)

        self.get_logger().info("State machine node initialized")
        self.get_logger().info(f"Output path for lanterns: {self.lantern_output_path}")
        self._publish_enable_local_planner(False)

    def _get_vec_param(self, name: str, optional: bool = False):
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            value = []
        if value is None or value == []:
            if optional:
                return None
            raise ValueError(f"Parameter '{name}' must be a list of 3 numbers")

        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Parameter '{name}' must be a list of 3 numbers")
        return [float(v) for v in value]

    def _resolve_output_path(self, path: str) -> str:
        if path:
            return os.path.expanduser(path)
        share_dir = get_package_share_directory("state_machine_pkg")
        return os.path.join(share_dir, "lantern_positions.json")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation
        self.current_yaw = self._yaw_from_quaternion(self.current_orientation)
        if not self.have_odom:
            self.have_odom = True
            self.get_logger().info("Received first odometry")

    def lantern_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            self.detected_lanterns[marker.id] = {
                "x": marker.pose.position.x,
                "y": marker.pose.position.y,
                "z": marker.pose.position.z,
            }

        if self.detected_lanterns:
            self._save_lantern_positions()

    def on_timer(self):
        if not self.have_odom:
            return

        if self.state == MissionState.IDLE:
            if self.auto_start:
                self._transition_to(MissionState.TAKEOFF)
            return

        if self.state == MissionState.TAKEOFF:
            target = [self.current_pose.x, self.current_pose.y, self.takeoff_alt]
            self._publish_trajectory(target)
            if self._is_at_target(target):
                self._transition_to(MissionState.GO_TO_ENTRANCE)
            return

        if self.state == MissionState.GO_TO_ENTRANCE:
            if not self.use_basic_waypoint_entry:
                self._publish_trajectory(self.entry_goal)
            if self._is_at_target(self.entry_goal):
                self._transition_to(MissionState.EXPLORE)
            return

        if self.state == MissionState.EXPLORE:
            self._publish_enable_local_planner(True)
            if self._explore_complete():
                self._transition_to(MissionState.LAND)
            return

        if self.state == MissionState.LAND:
            if self.land_target is None:
                self.land_target = self._compute_land_target()
            self._publish_trajectory(self.land_target)
            if self._is_at_target(self.land_target):
                self._transition_to(MissionState.DONE)
            return

        if self.state == MissionState.DONE:
            self._publish_enable_local_planner(False)

    def _transition_to(self, new_state: MissionState):
        if new_state == self.state:
            return

        self.get_logger().info(f"Transition: {self.state.value} -> {new_state.value}")
        self.state = new_state

        if new_state == MissionState.EXPLORE:
            self.explore_start_time = self.get_clock().now()
            self._publish_enable_local_planner(True)
        elif new_state in (MissionState.TAKEOFF, MissionState.GO_TO_ENTRANCE, MissionState.LAND, MissionState.DONE):
            self._publish_enable_local_planner(False)

        if new_state == MissionState.LAND:
            self._save_lantern_positions()

        if new_state == MissionState.DONE:
            self._save_lantern_positions()

    def _publish_enable_local_planner(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.enable_local_planner_pub.publish(msg)

    def _publish_trajectory(self, target):
        traj = MultiDOFJointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = "world"

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = float(target[0])
        transform.translation.y = float(target[1])
        transform.translation.z = float(target[2])
        transform.rotation = self._quaternion_from_yaw(self.current_yaw)
        point.transforms = [transform]
        point.velocities = [Twist()]
        point.accelerations = [Twist()]
        traj.points = [point]

        self.trajectory_pub.publish(traj)

    def _compute_land_target(self):
        if self.land_goal is not None:
            return self.land_goal
        return [self.current_pose.x, self.current_pose.y, self.land_z]

    def _is_at_target(self, target):
        dx = self.current_pose.x - target[0]
        dy = self.current_pose.y - target[1]
        dz = self.current_pose.z - target[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= self.goal_tolerance

    def _explore_complete(self):
        if self.target_lantern_count > 0 and len(self.detected_lanterns) >= self.target_lantern_count:
            self.get_logger().info(
                f"Lantern target reached: {len(self.detected_lanterns)} / {self.target_lantern_count}"
            )
            return True

        if self.explore_start_time is not None:
            elapsed = (self.get_clock().now() - self.explore_start_time).nanoseconds * 1e-9
            if elapsed >= self.explore_timeout_sec:
                self.get_logger().warn("Explore timeout reached, initiating landing")
                return True

        return False

    def _save_lantern_positions(self):
        try:
            directory = os.path.dirname(self.lantern_output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            data = {
                "count": len(self.detected_lanterns),
                "lanterns": [
                    {"id": k, **v} for k, v in sorted(self.detected_lanterns.items(), key=lambda item: item[0])
                ],
            }
            with open(self.lantern_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            self.get_logger().error(f"Failed to write lantern positions: {exc}")

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_from_yaw(yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
