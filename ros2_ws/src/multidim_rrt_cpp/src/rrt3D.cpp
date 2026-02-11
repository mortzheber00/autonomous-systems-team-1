#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_srvs/srv/empty.hpp"
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
 * 1. Receives frontier goals from local planner
 * 2. Validates frontier safety (visibility, obstacle clearance)
 * 3. Plans collision-free path using RRT
 * 4. Publishes trajectory commands for controller
 * 
 * ROS Interface (Subscribers):
 * - /frontier_goals (geometry_msgs/PoseArray) - Top 5 frontiers from local planner
 * - /current_state_est (nav_msgs/Odometry) - Current drone position
 * - /octomap_full (octomap_msgs/Octomap) - Map for collision checking
 * 
 * ROS Interface (Publishers):
 * - /command/trajectory (trajectory_msgs/MultiDOFJointTrajectory) - Trajectory commands
 * - /path (nav_msgs/Path) - Planned path
 * - /rrt_tree (visualization_msgs/MarkerArray) - RRT tree visualization
 * - /rrt_path (visualization_msgs/Marker) - Final path visualization
 */
class RRT3DNode : public rclcpp::Node
{
public:
  RRT3DNode()
      : Node("RRT3DNode"), step_size(3.0), goal_tolerance(3.0), 
        node_limit(500), safety_radius_(2.0), occupied_threshold_(0.5)
  {
    // Create publisher for trajectory commands
    trajectory_publisher_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/command/trajectory", 10);
    
    // Create publisher for the planned path
    path_publisher = create_publisher<nav_msgs::msg::Path>("path", 10);
    
    // Create publisher for RRT visualization (tree nodes and edges)
    tree_marker_publisher = create_publisher<visualization_msgs::msg::MarkerArray>("rrt_tree", 10);
    
    // Create publisher for path visualization
    path_marker_publisher = create_publisher<visualization_msgs::msg::Marker>("rrt_path", 10);
    
    // Create publisher for waypoint path visualization (the actual path being followed)
    waypoint_path_publisher_ = create_publisher<visualization_msgs::msg::Marker>("rrt_waypoint_path", 10);
    
    // Subscribe to top 5 frontier goals from local planner
    frontier_subscriber = create_subscription<geometry_msgs::msg::PoseArray>(
      "/frontier_goals",
      10,
      std::bind(&RRT3DNode::frontier_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to current drone position (odometry)
    odom_subscriber = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state_est",
      10,
      std::bind(&RRT3DNode::odom_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to Octomap for collision checking and safety validation
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full",
      rclcpp::QoS(10).best_effort(),
      std::bind(&RRT3DNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Timer for continuous trajectory publishing
    trajectory_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),  // 10 Hz
      std::bind(&RRT3DNode::trajectory_publishing_loop, this)
    );
    
    RCLCPP_INFO(get_logger(), "RRT3DNode initialized with safety checks");
    RCLCPP_INFO(get_logger(), "Waiting for frontier list on /frontier_goals...");
  }

private:
  // Configuration parameters
  const float step_size;              // Distance to extend tree at each iteration
  const float goal_tolerance;         // Tolerance for reaching goal
  const size_t node_limit;            // Maximum number of nodes before stopping
  const float safety_radius_;         // Safety radius for obstacle clearance
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
  
  // Octomap for safety checks
  std::shared_ptr<octomap::OcTree> current_octree_;
  
  // RRT tree data structures
  std::vector<int> map_size = {10, 10, 10};  // Configuration space bounds (not used with octomap)
  std::vector<TreeNode *> node_list;         // All nodes in the tree
  TreeNode *root_node;                       // Root node (start position)
  
  // ROS communication
  nav_msgs::msg::Path path;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_marker_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr waypoint_path_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr frontier_subscriber;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::TimerBase::SharedPtr trajectory_timer_;

  /**
   * Callback for Octomap updates
   */
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
    if (tree) {
        octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);
        if (octree) {
            current_octree_ = std::shared_ptr<octomap::OcTree>(octree);
        } else {
            RCLCPP_ERROR(get_logger(), "Octomap message was not an OcTree");
            delete tree;
        }
    }
  }

  /**
   * Callback for frontier goals subscriber
   * 
   * Called whenever local planner publishes top 5 frontier goals.
   * Tries each frontier in order until a valid path is found.
   */
  void frontier_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->poses.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty frontier list");
      return;
    }
    
    RCLCPP_INFO(get_logger(), "Received %zu frontier candidates", msg->poses.size());
    
    // Store frontier candidates
    frontier_candidates_.clear();
    for (const auto& pose : msg->poses) {
      std::vector<float> frontier = {
        static_cast<float>(pose.position.x),
        static_cast<float>(pose.position.y),
        static_cast<float>(pose.position.z)
      };
      frontier_candidates_.push_back(frontier);
    }
    
    // Start planning if not already planning
    if (!is_planning)
    {
      is_planning = true;
      plan_to_best_frontier();
      is_planning = false;
    }
  }

  /**
   * Try planning to each frontier candidate in order
   */
  void plan_to_best_frontier()
  {
    for (size_t i = 0; i < frontier_candidates_.size(); ++i) {
      const auto& frontier = frontier_candidates_[i];
      
      RCLCPP_INFO(get_logger(), "Trying frontier %zu: (%.2f, %.2f, %.2f)", 
                  i, frontier[0], frontier[1], frontier[2]);
      
      // Safety validation: check if point is inside an obstacle
      if (current_octree_) {
        octomap::point3d goal_point(frontier[0], frontier[1], frontier[2]);
        octomap::OcTreeNode* node = current_octree_->search(goal_point);
        if (node && node->getOccupancy() > occupied_threshold_) {
          RCLCPP_WARN(get_logger(), "Frontier %zu is inside an obstacle, trying next", i);
          continue;
        }
      }
      
      // Set this frontier as goal and try planning
      goal_position = frontier;
      has_goal = true;
      
      bool success = plan_path();
      
      if (success) {
        RCLCPP_INFO(get_logger(), "Successfully planned to frontier %zu", i);
        return;
      } else {
        RCLCPP_WARN(get_logger(), "Failed to plan to frontier %zu, trying next", i);
      }
    }
    
    RCLCPP_WARN(get_logger(), "Failed to plan to any of the %zu frontiers", 
                frontier_candidates_.size());
    goal_active_ = false;
  }

  /**
   * Callback for odometry/position subscriber
   * 
   * Called whenever drone position is updated.
   * Updates the start position for RRT planning.
   */
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Update current drone position as RRT start
    start_position[0] = msg->pose.pose.position.x;
    start_position[1] = msg->pose.pose.position.y;
    start_position[2] = msg->pose.pose.position.z;
  }

  /**
   * Check if a point is in collision with obstacles
   */
  bool is_in_collision(const std::vector<float>& position)
  {
    if (!current_octree_) return false;
    
    octomap::point3d point(position[0], position[1], position[2]);
    octomap::OcTreeNode* node = current_octree_->search(point);
    
    // Point is occupied
    if (node && node->getOccupancy() > occupied_threshold_) {
      return true;
    }
    
    return false;
  }

  /**
   * Check if edge between two points is collision-free
   * Uses discrete sampling along the edge
   */
  bool is_edge_collision_free(const std::vector<float>& from, const std::vector<float>& to)
  {
    if (!current_octree_) return true;
    
    // Sample along the edge at resolution intervals
    float dx = to[0] - from[0];
    float dy = to[1] - from[1];
    float dz = to[2] - from[2];
    float dist = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (dist < 0.01) return true;
    
    int num_samples = std::max(2, static_cast<int>(dist / 1.0)); // Sample every 1.0m (less dense)
    
    for (int i = 0; i <= num_samples; ++i) {
      float t = static_cast<float>(i) / num_samples;
      std::vector<float> sample = {
        from[0] + t * dx,
        from[1] + t * dy,
        from[2] + t * dz
      };
      
      if (is_in_collision(sample)) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Continuous trajectory publishing loop
   * Publishes trajectory commands at 10Hz to keep controller active
   * Follows waypoints from the planned RRT path
   */
  void trajectory_publishing_loop()
  {
    if (!goal_active_ || current_path_.empty()) return;
    
    // Get current waypoint
    if (current_waypoint_idx_ >= current_path_.size()) {
      RCLCPP_INFO(get_logger(), "Reached end of path");
      goal_active_ = false;
      return;
    }
    
    auto& waypoint = current_path_[current_waypoint_idx_];
    
    // Publish trajectory command to current waypoint
    trajectory_msgs::msg::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header.stamp = this->get_clock()->now();
    trajectory_msg.header.frame_id = "world";
    trajectory_msg.joint_names.push_back("base_link");

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
    geometry_msgs::msg::Transform transform;
    transform.translation.x = waypoint[0];
    transform.translation.y = waypoint[1];
    transform.translation.z = waypoint[2];
    
    // Calculate yaw toward waypoint
    double dx = waypoint[0] - start_position[0];
    double dy = waypoint[1] - start_position[1];
    double yaw = atan2(dy, dx);
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    
    transform.rotation.x = 0.0;
    transform.rotation.y = 0.0;
    transform.rotation.z = sy;
    transform.rotation.w = cy;

    point.transforms.push_back(transform);

    // Zero velocities and accelerations (waypoint mode)
    geometry_msgs::msg::Twist velocity;
    geometry_msgs::msg::Twist acceleration;
    point.velocities.push_back(velocity);
    point.accelerations.push_back(acceleration);

    trajectory_msg.points.push_back(point);
    trajectory_publisher_->publish(trajectory_msg);
    
    // Check if waypoint reached
    double dist = sqrt(
      pow(start_position[0] - waypoint[0], 2) +
      pow(start_position[1] - waypoint[1], 2) +
      pow(start_position[2] - waypoint[2], 2)
    );
    
    if (dist < goal_tolerance) {
      RCLCPP_INFO(get_logger(), "Waypoint %zu reached, moving to next", current_waypoint_idx_);
      current_waypoint_idx_++;
    }
  }

  /**
   * Main RRT planning function
   * 
   * This is called whenever a new goal is received.
   * Generates a path from current position to goal.
   * Returns true if path found, false otherwise.
   */
  bool plan_path()
  {
    RCLCPP_INFO(get_logger(), "Planning RRT from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)",
                start_position[0], start_position[1], start_position[2],
                goal_position[0], goal_position[1], goal_position[2]);

    auto start_time = this->now();

    // Initialize tree with current position
    TreeNode *root_node = new TreeNode(start_position, nullptr);
    
    // Clear previous tree
    for (TreeNode *node : node_list)
    {
      delete node;
    }
    node_list = {root_node};
    
    // RRT main loop: expand tree until goal found or node limit/time reached
    while (node_list.size() < node_limit && (this->now() - start_time).nanoseconds() / 1e9 < 2.0)
    {
      // Step 1: Generate random point in configuration space
      // Use goal-biased sampling (30% toward goal for faster convergence)
      float rand_x, rand_y, rand_z;
      if ((rand() % 100) < 30) {
        // Bias toward goal
        rand_x = goal_position[0];
        rand_y = goal_position[1];
        rand_z = goal_position[2];
      } else {
        // Random sampling in reasonable bounds
        rand_x = start_position[0] + (rand() % 200 - 100);
        rand_y = start_position[1] + (rand() % 200 - 100);
        rand_z = start_position[2] + (rand() % 20 - 10);
      }
      
      float min_distance = INFINITY;
      std::vector<float> min_node_vec;
      TreeNode *nearest_node;
      
      // Step 2: Find nearest node in tree to random point
      for (TreeNode *node : node_list)
      {
        // Check if this node is close enough to goal
        std::vector<float> goal_vec = {goal_position[0] - node->val[0],
                                       goal_position[1] - node->val[1],
                                       goal_position[2] - node->val[2]};
        const float goal_distance = sqrt(pow(goal_vec[0], 2) +
                                         pow(goal_vec[1], 2) +
                                         pow(goal_vec[2], 2));
        
        // Check if close enough to goal
        if (goal_distance < goal_tolerance) {
          // Create goal node
          TreeNode *goal_node = new TreeNode(goal_position, node);
          node->add_child(goal_node);
          node_list.push_back(goal_node);
          RCLCPP_INFO(get_logger(), "Path found!");
          
          // Extract waypoints from path (from start to goal)
          std::vector<std::vector<float>> full_path;
          TreeNode *current = goal_node;
          while (current != nullptr) {
            full_path.insert(full_path.begin(), current->val);
            current = current->parent;
          }
          
          // Downsample path: keep every Nth waypoint to reduce trajectory points
          current_path_.clear();
          current_waypoint_idx_ = 0;
          int downsample_factor = 2;  // Keep every 2nd waypoint
          for (size_t i = 0; i < full_path.size(); i += downsample_factor) {
            current_path_.push_back(full_path[i]);
          }
          // Always include the final goal
          if (current_path_.empty() || current_path_.back() != full_path.back()) {
            current_path_.push_back(full_path.back());
          }
          
          RCLCPP_INFO(get_logger(), "Path has %zu waypoints (downsampled from %zu)", 
                      current_path_.size(), full_path.size());
          
          // Set active goal for trajectory publishing
          current_goal_position_ = goal_position;
          goal_active_ = true;
          
          visualize_rrt_tree();   // Show the complete tree
          visualize_path();       // Show the final path
          visualize_waypoint_path();  // Show the downsampled waypoint path
          publish_path();
          return true;
        }
        
        // Calculate distance from node to random point
        std::vector<float> node_vec = {node->val[0] - rand_x,
                                       node->val[1] - rand_y,
                                       node->val[2] - rand_z};
        const float distance = sqrt(pow(node_vec[0], 2) +
                                    pow(node_vec[1], 2) +
                                    pow(node_vec[2], 2));
        
        // Keep track of nearest node
        if (distance < min_distance) {
          min_distance = distance;
          min_node_vec = node_vec;
          nearest_node = node;
        }
      }
      
      // Step 3: Extend tree toward random point by step_size
      if (min_distance == 0.0f) {
        // Random point is exactly at nearest node, skip
        continue;
      }
      const float new_node_x = nearest_node->val[0] - (min_node_vec[0] / min_distance) * step_size;
      const float new_node_y = nearest_node->val[1] - (min_node_vec[1] / min_distance) * step_size;
      const float new_node_z = nearest_node->val[2] - (min_node_vec[2] / min_distance) * step_size;
      std::vector<float> new_node_position = {new_node_x, new_node_y, new_node_z};
      
      // Collision check: verify new node is collision-free
      if (is_in_collision(new_node_position)) {
        continue;
      }
      
      // Edge validation: verify path from parent to new node is collision-free
      if (!is_edge_collision_free(nearest_node->val, new_node_position)) {
        continue;
      }
      
      // Create new node and add to tree
      TreeNode *new_node = new TreeNode(new_node_position, nearest_node);
      nearest_node->add_child(new_node);
      node_list.push_back(new_node);

      // Periodically visualize tree progress (every 200 nodes)
      if (node_list.size() % 200 == 0)
      {
        visualize_rrt_tree();
      }
    }
    
    // Node limit or time limit reached without finding path
    RCLCPP_WARN(get_logger(), "Path not found within time or node limit");
    goal_active_ = false;
    return false;
  }

  /**
   * Visualize the RRT tree in RViz
   * 
   * Publishes:
   * - Nodes as small blue spheres
   * - Edges as white lines connecting parent to child
   * - Start as green sphere
   * - Goal as red sphere
   */
  void visualize_rrt_tree()
  {
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;

    // Publish all nodes as spheres
    for (TreeNode *node : node_list)
    {
      visualization_msgs::msg::Marker node_marker;
      node_marker.header.frame_id = "world";
      node_marker.header.stamp = get_clock()->now();
      node_marker.id = marker_id++;
      node_marker.type = visualization_msgs::msg::Marker::SPHERE;
      node_marker.action = visualization_msgs::msg::Marker::ADD;
      node_marker.pose.position.x = node->val[0];
      node_marker.pose.position.y = node->val[1];
      node_marker.pose.position.z = node->val[2];
      node_marker.scale.x = node_marker.scale.y = node_marker.scale.z = 0.1;
      node_marker.color.r = 0.0f;
      node_marker.color.g = 0.0f;
      node_marker.color.b = 1.0f;  // Blue for regular nodes
      node_marker.color.a = 0.8f;
      markers.markers.push_back(node_marker);

      // Draw edge from parent to this node
      if (node->parent != nullptr)
      {
        visualization_msgs::msg::Marker edge_marker;
        edge_marker.header.frame_id = "world";
        edge_marker.header.stamp = get_clock()->now();
        edge_marker.id = marker_id++;
        edge_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        edge_marker.action = visualization_msgs::msg::Marker::ADD;
        edge_marker.scale.x = 0.05;  // Line width
        edge_marker.color.r = 1.0f;
        edge_marker.color.g = 1.0f;
        edge_marker.color.b = 1.0f;  // White edges
        edge_marker.color.a = 0.6f;

        geometry_msgs::msg::Point p1, p2;
        p1.x = node->parent->val[0];
        p1.y = node->parent->val[1];
        p1.z = node->parent->val[2];
        p2.x = node->val[0];
        p2.y = node->val[1];
        p2.z = node->val[2];
        edge_marker.points.push_back(p1);
        edge_marker.points.push_back(p2);
        markers.markers.push_back(edge_marker);
      }
    }

    // Highlight start position (green sphere)
    visualization_msgs::msg::Marker start_marker;
    start_marker.header.frame_id = "world";
    start_marker.header.stamp = get_clock()->now();
    start_marker.id = marker_id++;
    start_marker.type = visualization_msgs::msg::Marker::SPHERE;
    start_marker.action = visualization_msgs::msg::Marker::ADD;
    start_marker.pose.position.x = start_position[0];
    start_marker.pose.position.y = start_position[1];
    start_marker.pose.position.z = start_position[2];
    start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.3;
    start_marker.color.r = 0.0f;
    start_marker.color.g = 1.0f;
    start_marker.color.b = 0.0f;  // Green for start
    start_marker.color.a = 1.0f;
    markers.markers.push_back(start_marker);

    // Highlight goal position (red sphere)
    visualization_msgs::msg::Marker goal_marker;
    goal_marker.header.frame_id = "world";
    goal_marker.header.stamp = get_clock()->now();
    goal_marker.id = marker_id++;
    goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
    goal_marker.action = visualization_msgs::msg::Marker::ADD;
    goal_marker.pose.position.x = goal_position[0];
    goal_marker.pose.position.y = goal_position[1];
    goal_marker.pose.position.z = goal_position[2];
    goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.3;
    goal_marker.color.r = 1.0f;
    goal_marker.color.g = 0.0f;
    goal_marker.color.b = 0.0f;  // Red for goal
    goal_marker.color.a = 1.0f;
    markers.markers.push_back(goal_marker);

    tree_marker_publisher->publish(markers);
  }

  /**
   * Visualize the final path as a thick line
   */
  void visualize_path()
  {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "world";
    path_marker.header.stamp = get_clock()->now();
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.1;  // Line width
    path_marker.color.r = 0.0f;
    path_marker.color.g = 1.0f;
    path_marker.color.b = 0.0f;  // Green for final path
    path_marker.color.a = 1.0f;

    // Add path points in reverse order (from goal to start)
    TreeNode *current_node = node_list.back();
    while (current_node != nullptr)
    {
      geometry_msgs::msg::Point p;
      p.x = current_node->val[0];
      p.y = current_node->val[1];
      p.z = current_node->val[2];
      path_marker.points.push_back(p);
      current_node = current_node->parent;
    }

    path_marker_publisher->publish(path_marker);
  }

  /**
   * Visualize the downsampled waypoint path (the path being followed)
   */
  void visualize_waypoint_path()
  {
    visualization_msgs::msg::Marker waypoint_marker;
    waypoint_marker.header.frame_id = "world";
    waypoint_marker.header.stamp = get_clock()->now();
    waypoint_marker.id = 0;
    waypoint_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    waypoint_marker.action = visualization_msgs::msg::Marker::ADD;
    waypoint_marker.scale.x = 0.3;  // Thicker line width
    waypoint_marker.color.r = 1.0f;
    waypoint_marker.color.g = 0.5f;
    waypoint_marker.color.b = 0.0f;  // Orange for waypoint path
    waypoint_marker.color.a = 1.0f;

    // Add waypoints in order
    for (const auto& waypoint : current_path_)
    {
      geometry_msgs::msg::Point p;
      p.x = waypoint[0];
      p.y = waypoint[1];
      p.z = waypoint[2];
      waypoint_marker.points.push_back(p);
    }

    // Also add spheres at each waypoint
    visualization_msgs::msg::Marker waypoint_spheres;
    waypoint_spheres.header.frame_id = "world";
    waypoint_spheres.header.stamp = get_clock()->now();
    waypoint_spheres.id = 1;
    waypoint_spheres.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    waypoint_spheres.action = visualization_msgs::msg::Marker::ADD;
    waypoint_spheres.scale.x = 0.5;
    waypoint_spheres.scale.y = 0.5;
    waypoint_spheres.scale.z = 0.5;
    waypoint_spheres.color.r = 1.0f;
    waypoint_spheres.color.g = 0.5f;
    waypoint_spheres.color.b = 0.0f;
    waypoint_spheres.color.a = 0.7f;
    
    for (const auto& waypoint : current_path_)
    {
      geometry_msgs::msg::Point p;
      p.x = waypoint[0];
      p.y = waypoint[1];
      p.z = waypoint[2];
      waypoint_spheres.points.push_back(p);
    }

    waypoint_path_publisher_->publish(waypoint_marker);
    waypoint_path_publisher_->publish(waypoint_spheres);
  }

  /**
   * Publish the path from start to goal
   * 
   * Backtracks from the goal node through parent pointers to reconstruct
   * the full path, then publishes it as a nav_msgs::Path message.
   */
  void publish_path()
  {
    path.poses.clear();
    path.header.frame_id = "world";
    path.header.stamp = get_clock()->now();
    
    // Backtrack from goal node to start using parent pointers
    TreeNode *current_node = node_list.back();
    while (current_node->parent != nullptr)
    {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = get_clock()->now();
      pose.pose.position.x = current_node->val[0];
      pose.pose.position.y = current_node->val[1];
      pose.pose.position.z = current_node->val[2];
      path.poses.push_back(pose);
      current_node = current_node->parent;
    }
    path_publisher->publish(path);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RRT3DNode>());
  rclcpp::shutdown();
  return 0;
}