#ifndef RRT3D_HPP
#define RRT3D_HPP

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_srvs/srv/empty.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include "TreeNode.h"
#include <memory>

/**
 * RRT3DNode - 3D Rapid Exploring Random Tree Planner with Safety Checks
 * 
 * This ROS 2 node implements the RRT algorithm for safe 3D path planning.
 * 
 * Algorithm Overview:
 * 1. Receives frontier goals from local planner OR backtrack goals from global planner
 * 2. Validates frontier safety (visibility, obstacle clearance)
 * 3. Plans collision-free path using RRT
 * 4. Publishes trajectory commands for controller
 * 5. For backtrack mode: plans through each waypoint sequentially, signals completion
 * 
 * ROS Interface (Subscribers):
 * - /frontier_goals (geometry_msgs/PoseArray) - Top 5 frontiers from local planner
 * - /backtrack_goals (geometry_msgs/PoseArray) - Backtrack waypoints from global planner
 * - /current_state_est (nav_msgs/Odometry) - Current drone position
 * - /octomap_full (octomap_msgs/Octomap) - Map for collision checking
 * 
 * ROS Interface (Publishers):
 * - /command/trajectory (trajectory_msgs/MultiDOFJointTrajectory) - Trajectory commands
 * - /backtrack_complete (std_msgs/Bool) - Signal when backtrack sequence is done
 * - /path (nav_msgs/Path) - Planned path
 * - /rrt_tree (visualization_msgs/MarkerArray) - RRT tree visualization
 * - /rrt_path (visualization_msgs/Marker) - Final path visualization
 */
class RRT3DNode : public rclcpp::Node
{
public:
  RRT3DNode();

private:
  // Configuration parameters
  const float step_size;              // Distance to extend tree at each iteration
  const float goal_tolerance;         // Tolerance for reaching goal
  const size_t node_limit;            // Maximum number of nodes before stopping
  const float safety_radius_;         // Safety radius for RRT tree nodes and edges (keeps drone safe)
  const float frontier_safety_radius_; // Relaxed safety radius for frontier goal validation
  const float occupied_threshold_;    // Occupancy probability threshold
  
  // Current state (updated by subscribers)
  std::vector<float> start_position = {0.0f, 0.0f, 0.0f};  // Current drone position
  std::vector<float> goal_position = {5.0f, 5.0f, 5.0f};   // Goal from local planner
  std::vector<std::vector<float>> frontier_candidates_;     // List of frontier candidates
  std::vector<float> current_goal_position_;                // Active goal for trajectory publishing
  std::vector<std::vector<float>> current_path_;            // Waypoints from RRT path
  size_t current_waypoint_idx_ = 0;                         // Index of current waypoint in path
  bool has_goal = false;                                    // Whether we have a valid goal
  bool is_planning = false;                                 // Prevent concurrent planning
  bool goal_active_ = false;                                // Whether a goal is currently active
  
  // Backtrack mode state
  bool backtrack_mode_ = false;                             // Whether we're in backtrack mode
  std::vector<std::vector<float>> backtrack_waypoints_;     // Full list of backtrack waypoints
  size_t backtrack_waypoint_idx_ = 0;                       // Current backtrack waypoint being targeted
  
  // Octomap for safety checks
  std::shared_ptr<octomap::OcTree> current_octree_;
  
  // RRT tree data structures
  std::vector<int> map_size = {10, 10, 10};  // Configuration space bounds (not used with octomap)
  std::vector<TreeNode *> node_list;         // All nodes in the tree
  TreeNode *root_node;                       // Root node (start position)
  
  // ROS communication
  nav_msgs::msg::Path path;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr backtrack_complete_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_marker_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr waypoint_path_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr frontier_subscriber;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr backtrack_subscriber;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::TimerBase::SharedPtr trajectory_timer_;

  // Callback methods
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg);
  void frontier_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void backtrack_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void trajectory_publishing_loop();

  // Planning methods
  void plan_to_best_frontier();
  void plan_backtrack_sequence();
  bool plan_path();

  // Safety and collision checking
  bool is_in_collision(const std::vector<float>& position);
  bool is_in_collision(const std::vector<float>& position, float radius);
  bool is_edge_collision_free(const std::vector<float>& from, const std::vector<float>& to);

  // Visualization methods
  void visualize_rrt_tree();
  void visualize_path();
  void visualize_waypoint_path();
  void publish_path();
};

#endif // RRT3D_HPP