#!/usr/bin/env python3

import math
import select
import sys
import termios
import tty
from math import cos, sin

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Transform, Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')
        
        # Initialize current state
        self.current_pose = Point(x=0.0, y=0.0, z=1.0)
        self.current_heading = 0.0
        
        # Create publisher and subscriber
        self.trajectory_pub = self.create_publisher(
            MultiDOFJointTrajectory,
            '/command/trajectory',
            10
        )
        self.pose_sub = self.create_subscription(
            Odometry,
            '/current_state',
            self.pose_callback,
            10
        )
        
        self.get_logger().info('Keyboard control node initialized')

    # --- Math Helper Functions (Replacements for tf.transformations) ---
    def get_yaw_from_quaternion(self, q: Quaternion) -> float:
        """
        Calculate yaw from a geometry_msgs Quaternion.
        """
        # yaw (z-axis rotation)
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(t3, t4)

    def get_quaternion_from_yaw(self, yaw: float) -> Quaternion:
        """
        Create a geometry_msgs Quaternion from yaw (assuming roll=0, pitch=0).
        """
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = sin(yaw / 2.0)
        q.w = cos(yaw / 2.0)
        return q
    # -----------------------------------------------------------------

    def pose_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position
        self.current_heading = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

    def get_key(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            # Check for input with a small timeout so we don't block the loop entirely
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def key_to_trajectory(self, key: str) -> MultiDOFJointTrajectory:
        step_size = 3.0
        rotation_step_size = 0.1
        pose = [self.current_pose.x, self.current_pose.y, self.current_pose.z]
        desired_yaw = self.current_heading

        trajectory = MultiDOFJointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = "world"

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()

        if key == "w":
            pose[0] += step_size * cos(self.current_heading)
            pose[1] += step_size * sin(self.current_heading)
        elif key == "s":
            pose[0] -= step_size * cos(self.current_heading)
            pose[1] -= step_size * sin(self.current_heading)
        elif key == "a":
            pose[0] -= step_size * sin(self.current_heading)
            pose[1] += step_size * cos(self.current_heading)
        elif key == "d":
            pose[0] += step_size * sin(self.current_heading)
            pose[1] -= step_size * cos(self.current_heading)
        elif key == "q":
            pose[2] += step_size
        elif key == "e":
            pose[2] -= step_size
        elif key == "j":
            desired_yaw += rotation_step_size
        elif key == "l":
            desired_yaw -= rotation_step_size

        transform.translation.x = pose[0]
        transform.translation.y = pose[1]
        transform.translation.z = pose[2]
        
        # Use helper function to create quaternion
        transform.rotation = self.get_quaternion_from_yaw(desired_yaw)
        
        point.transforms = [transform]
        point.velocities = [Twist()]
        point.accelerations = [Twist()]
        trajectory.points.append(point)
        return trajectory

    def run(self):
        print("\nKeyboard Control:")
        print("  w/s: forward/backward")
        print("  a/d: left/right")
        print("  q/e: up/down")
        print("  j/l: rotate left/right")
        print("  p: quit")
        print("\nPress keys on the keyboard:")
        
        try:
            while rclpy.ok():
                # 1. Read key (non-blocking-ish due to select timeout)
                key = self.get_key()
                
                if key == 'p':
                    self.get_logger().info('Quitting...')
                    break
                elif key and key in 'wsadqejl':
                    trajectory = self.key_to_trajectory(key)
                    self.trajectory_pub.publish(trajectory)
                    self.get_logger().info(f'Published trajectory for key: {key}')
                
                # 2. Process ROS callbacks (like odometry)
                rclpy.spin_once(self, timeout_sec=0.01)
                
        except KeyboardInterrupt:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    
    try:
        node.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()