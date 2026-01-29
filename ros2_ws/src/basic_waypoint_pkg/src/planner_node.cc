/*
 * Simple example that shows a trajectory planner using
 * mav_trajectory_generation in ROS2.
 *
 * Build + run via your ROS2 launch setup, e.g.
 *   ros2 run basic_waypoint_pkg basic_waypoint_node
 */

 #include <memory>
 #include <iostream>
 
 #include "rclcpp/rclcpp.hpp"
 #include "Eigen/Dense"
 
 #include "basic_waypoint_pkg/planner.hpp"
 
 int main(int argc, char ** argv)
 {
   rclcpp::init(argc, argv);
 
   // Create node
   auto node = rclcpp::Node::make_shared("simple_planner");
 
   // Instantiate basic planner
   BasicPlanner planner(node);
 
   // Wait for initialization (odometry)
   rclcpp::Rate loop_rate(10.0);
   while(rclcpp::ok() && !planner.isInitialized()) {
       rclcpp::spin_some(node);
       RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "Waiting for Odometry...");
       loop_rate.sleep();
   }
 
   // Define goal point
   Eigen::Vector3d goal_position, goal_velocity;
   goal_position << -360.0, 0.0, 16.0; //12.0, 39.0, 9.0
   goal_velocity << 0.0, 0.0, 0.0;
 
   // Plan and publish trajectory
   mav_trajectory_generation::Trajectory trajectory;
   planner.planTrajectory(goal_position, goal_velocity, &trajectory);
   planner.publishTrajectory(trajectory);
 
   RCLCPP_WARN(node->get_logger(), "DONE. GOODBYE.");
 
   rclcpp::shutdown();
   return 0;
 }
 