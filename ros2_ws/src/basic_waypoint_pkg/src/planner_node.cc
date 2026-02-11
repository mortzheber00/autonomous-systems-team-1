/*
 * Simple example that shows a trajectory planner using
 * mav_trajectory_generation in ROS2.
 *
 * Build + run via your ROS2 launch setup, e.g.
 *   ros2 run basic_waypoint_pkg basic_waypoint_node
 */

 #include <memory>
 #include <iostream>
 #include <chrono>
 
 #include "rclcpp/rclcpp.hpp"
 #include "Eigen/Dense"
 
 #include "basic_waypoint_pkg/planner.hpp"
 
 int main(int argc, char ** argv)
 {
   rclcpp::init(argc, argv);
 
   // Create node - use "planner" to match config file namespace
   auto node = rclcpp::Node::make_shared("planner");
 
   // Instantiate basic planner
   BasicPlanner planner(node);
 
   // Wait for initialization (odometry)
   rclcpp::Rate loop_rate(10.0);
   while(rclcpp::ok() && !planner.isInitialized()) {
       rclcpp::spin_some(node);
       RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "Waiting for Odometry...");
       loop_rate.sleep();
   }
 
  // Define goal point (from parameters)
  std::vector<double> goal_flat;
  node->declare_parameter("goal", std::vector<double>());
  node->get_parameter("goal", goal_flat);
  if (goal_flat.size() != 3) {
    RCLCPP_ERROR(node->get_logger(), "Goal parameter must be a list of 3 numbers.");
    rclcpp::shutdown();
    return 1;
  }

  Eigen::Vector3d goal_position, goal_velocity;
  goal_position << goal_flat[0], goal_flat[1], goal_flat[2];
  goal_velocity << 0.0, 0.0, 0.0;
 
  // Plan and publish trajectory
  auto trajectory = std::make_shared<mav_trajectory_generation::Trajectory>();
  planner.planTrajectory(goal_position, goal_velocity, trajectory.get());
  planner.publishTrajectory(*trajectory);

  auto publish_timer = node->create_wall_timer(
    std::chrono::seconds(1),
    [trajectory, &planner]() {
      planner.publishTrajectory(*trajectory);
    });

  RCLCPP_WARN(node->get_logger(), "Publishing trajectory continuously.");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
 }
 