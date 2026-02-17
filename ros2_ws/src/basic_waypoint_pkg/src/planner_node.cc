/*
 * Trajectory planner using mav_trajectory_generation in ROS2.
 *
 * Accepts dynamic goals on the "goal_path" topic (nav_msgs/Path).
 * The last pose in the path is the goal; preceding poses are
 * intermediate waypoints.  An empty path cancels republishing.
 *
 * If the "goal" parameter is set (list of 3 doubles) the node will
 * also plan once at startup for backward-compatibility.
 */

#include <memory>
#include <iostream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "Eigen/Dense"
#include "nav_msgs/msg/path.hpp"

#include "basic_waypoint_pkg/planner.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Create node – name "planner" to match config-file namespace
  auto node = rclcpp::Node::make_shared("planner");

  // Instantiate basic planner
  BasicPlanner planner(node);

  // Wait for initialization (odometry)
  rclcpp::Rate loop_rate(10.0);
  while (rclcpp::ok() && !planner.isInitialized()) {
    rclcpp::spin_some(node);
    RCLCPP_INFO_THROTTLE(
      node->get_logger(), *node->get_clock(), 1000, "Waiting for Odometry...");
    loop_rate.sleep();
  }

  // Shared state for the current trajectory
  auto trajectory = std::make_shared<mav_trajectory_generation::Trajectory>();
  // Republish counter: publish a few times so late-joining subscribers can
  // catch the trajectory, then stop so the sampler isn't constantly reset.
  auto republish_remaining = std::make_shared<int>(0);

  // ------------------------------------------------------------------
  // (Optional) Plan once from the "goal" parameter for backward compat
  // ------------------------------------------------------------------
  std::vector<double> goal_flat;
  node->declare_parameter("goal", std::vector<double>());
  node->get_parameter("goal", goal_flat);

  if (goal_flat.size() == 3) {
    Eigen::Vector3d goal_position(goal_flat[0], goal_flat[1], goal_flat[2]);
    Eigen::Vector3d goal_velocity(0.0, 0.0, 0.0);
    if (planner.planTrajectory(goal_position, goal_velocity, trajectory.get())) {
      planner.publishTrajectory(*trajectory);
      *republish_remaining = 3;
      RCLCPP_INFO(node->get_logger(), "Planned initial trajectory from goal parameter.");
    }
  } else {
    RCLCPP_INFO(
      node->get_logger(),
      "No goal parameter provided. Waiting for goals on 'goal_path' topic...");
  }

  // ------------------------------------------------------------------
  // Subscribe to "goal_path" for dynamic goals from the state machine
  // ------------------------------------------------------------------
  auto goal_sub = node->create_subscription<nav_msgs::msg::Path>(
    "goal_path", 10,
    [&planner, trajectory, republish_remaining, &node](
      const nav_msgs::msg::Path::SharedPtr msg)
    {
      if (!planner.isInitialized()) {
        RCLCPP_WARN(node->get_logger(), "Planner not initialised \u2013 ignoring goal.");
        return;
      }

      // Empty path \u2192 cancel current trajectory republishing
      if (msg->poses.empty()) {
        *republish_remaining = 0;
        RCLCPP_INFO(node->get_logger(), "Received empty goal path – stopping republish.");
        return;
      }

      // Last pose = goal, preceding poses = intermediate waypoints
      std::vector<Eigen::Vector3d> waypoints;
      for (size_t i = 0; i + 1 < msg->poses.size(); ++i) {
        const auto & p = msg->poses[i].pose.position;
        waypoints.emplace_back(p.x, p.y, p.z);
      }
      planner.setWaypoints(waypoints);

      const auto & gp = msg->poses.back().pose.position;
      Eigen::Vector3d goal(gp.x, gp.y, gp.z);
      Eigen::Vector3d goal_vel(0.0, 0.0, 0.0);

      if (planner.planTrajectory(goal, goal_vel, trajectory.get())) {
        planner.publishTrajectory(*trajectory);
        *republish_remaining = 3;  // allow a few republishes for reliability
        RCLCPP_INFO(
          node->get_logger(),
          "Planned trajectory to (%.1f, %.1f, %.1f) with %zu waypoints.",
          goal.x(), goal.y(), goal.z(), waypoints.size());
      } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to plan trajectory to received goal.");
      }
    });

  // ------------------------------------------------------------------
  // Republish timer (1 Hz) so late-joining subscribers get the trajectory
  // ------------------------------------------------------------------
  auto publish_timer = node->create_wall_timer(
    std::chrono::seconds(1),
    [&planner, trajectory, republish_remaining]() {
      if (*republish_remaining > 0) {
        planner.publishTrajectory(*trajectory);
        --(*republish_remaining);
      }
    });

  RCLCPP_INFO(node->get_logger(), "Planner ready.");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
 