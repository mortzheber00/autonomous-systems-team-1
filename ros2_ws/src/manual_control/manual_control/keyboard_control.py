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
        
        # Current state (continuously updated from odometry)
        self.current_pose = Point(x=-38.0, y=10.0, z=9.0)
        self.current_heading = 0.0
        
        # Desired state (internal counter, only updated on valid key presses)
        self.desired_pose = Point(x=-38.0, y=10.0, z=9.0)
        self.desired_heading = 0.0
        
        # Flag to track if initial position has been read
        self.initial_pose_set = False
        
        # Threshold: only accept new inputs if drone is within this distance
        self.proximity_threshold = 9.0  # meters
        
        # Create publisher and subscriber
        self.trajectory_pub = self.create_publisher(
            MultiDOFJointTrajectory,
            '/command/trajectory',
            10
        )
        self.pose_sub = self.create_subscription(
            Odometry,
            '/current_state_est',
            self.pose_callback,
            10
        )
        
        self.get_logger().info('Keyboard control node initialized')
        self.get_logger().info('Waiting for initial odometry...')

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
        
        # Set initial desired state only once
        if not self.initial_pose_set:
            self.desired_pose = Point(
                x=self.current_pose.x,
                y=self.current_pose.y,
                z=self.current_pose.z
            )
            self.desired_heading = self.current_heading
            self.initial_pose_set = True
            self.get_logger().info(
                f'Initial position set: [{self.desired_pose.x:.2f}, '
                f'{self.desired_pose.y:.2f}, {self.desired_pose.z:.2f}]'
            )
    
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
    
    def distance_to_desired(self) -> float:
        """Calculate Euclidean distance from current pose to desired pose."""
        dx = self.current_pose.x - self.desired_pose.x
        dy = self.current_pose.y - self.desired_pose.y
        dz = self.current_pose.z - self.desired_pose.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def key_to_trajectory(self, key: str) -> MultiDOFJointTrajectory:
        step_size = 3.0
        rotation_step_size = 0.1
        
        # Work with internal desired state (not current_pose!)
        pose = [self.desired_pose.x, self.desired_pose.y, self.desired_pose.z]
        desired_yaw = self.desired_heading

        traj = MultiDOFJointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = 'world'

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()

        if key == "w":
            pose[0] += step_size * cos(self.desired_heading)
            pose[1] += step_size * sin(self.desired_heading)
        elif key == "s":
            pose[0] -= step_size * cos(self.desired_heading)
            pose[1] -= step_size * sin(self.desired_heading)
        elif key == "a":
            pose[0] -= step_size * sin(self.desired_heading)
            pose[1] += step_size * cos(self.desired_heading)
        elif key == "d":
            pose[0] += step_size * sin(self.desired_heading)
            pose[1] -= step_size * cos(self.desired_heading)
        elif key == "q":
            pose[2] += step_size
        elif key == "e":
            pose[2] -= step_size
        elif key == "j":
            desired_yaw += rotation_step_size
        elif key == "l":
            desired_yaw -= rotation_step_size

        # Update internal desired state
        self.desired_pose.x = pose[0]
        self.desired_pose.y = pose[1]
        self.desired_pose.z = pose[2]
        self.desired_heading = desired_yaw

        transform.translation.x = pose[0]
        transform.translation.y = pose[1]
        transform.translation.z = pose[2]
        
        # Use helper function to create quaternion
        transform.rotation = self.get_quaternion_from_yaw(desired_yaw)
        
        point.transforms = [transform]
        point.velocities = [Twist()]
        point.accelerations = [Twist()]
        
        traj.points.append(point)
        return traj

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
                # Process ROS callbacks to update current_pose
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # Wait for initial position to be set
                if not self.initial_pose_set:
                    continue
                
                # Read key (non-blocking)
                key = self.get_key()
                
                if key == 'p':
                    self.get_logger().info('Quitting...')
                    break
                elif key and key in 'wsadqejl':
                    # Check if drone is close enough to desired position
                    dist = self.distance_to_desired()
                    
                    if dist > self.proximity_threshold:
                        ##self.get_logger().warn(
                        ##    f'Drone too far from target ({dist:.2f}m > {self.proximity_threshold}m). '
                        ##    f'Wait for drone to reach waypoint first!'
                        ##)
                        continue
                    
                    # Generate and publish new trajectory
                    trajectory = self.key_to_trajectory(key)
                    self.trajectory_pub.publish(trajectory)
                    self.get_logger().info(
                        f'Key: {key} -> New target: [{self.desired_pose.x:.2f}, '
                        f'{self.desired_pose.y:.2f}, {self.desired_pose.z:.2f}] '
                        f'(current dist: {dist:.2f}m)'
                    )
                
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