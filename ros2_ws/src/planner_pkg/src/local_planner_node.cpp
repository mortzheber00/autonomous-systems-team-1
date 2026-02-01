/**
 * @file local_planner_node.cpp written by Phil Chudalla
 * @brief Frontier-Based Exploration Local Planner with Global Backtracking
 * This ROS 2 node implements a local planner that:
 * 1. Subscribes to Octomap updates from the mapping system
 * 2. Identifies frontier cells (boundaries between explored and unexplored space)
 * 3. Calculates exploration gain for each frontier
 * 4. Publishes the frontier with highest gain for the trajectory planner to execute
 * 5. Tracks flight history and backtracks to unexplored frontiers when none are found locally
 */

#include <rclcpp/rclcpp.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

/**
 * LocalPlannerNode - Frontier-Based Exploration Local Planner
 * Key improvements:
 * 1. Goal Commitment: Stick to a goal until reached or blocked.
 * 2. Constant Trajectory Publishing: Keeps controller active.
 * 3. Frontier-based exploration with global planner backtracking.
 */
class LocalPlannerNode : public rclcpp::Node
{
public:
  LocalPlannerNode() : Node("local_planner_node")
  {
    // --- Publishers -----
    // Create publisher for frontier goals
    frontier_publisher_ = create_publisher<geometry_msgs::msg::Point>("/frontier_goal", 10);
    marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>("/frontier_goal_marker", 10);
    trajectory_publisher_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/command/trajectory", 10);
    
    // Create publisher for frontier candidates (for global planner)
    frontier_candidates_publisher_ = create_publisher<nav_msgs::msg::Path>("/frontier_candidates", 10);
    
    // ----- Subscribers -----
    // Subscribe to Octomap updates and bind it to octomap callback (does frontier evaluation)
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full",
      rclcpp::QoS(10).best_effort(), // Best effort often better for heavy map data
      std::bind(&LocalPlannerNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Subscriber: Odometry
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state", 10,
      std::bind(&LocalPlannerNode::odom_callback, this, std::placeholders::_1)
    );

    // Timer: Run planning loop at 5 Hz (200ms)
    // This stabilizes the behavior by not replanning on every single map voxel change.
    planning_timer_ = create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&LocalPlannerNode::planning_loop, this)
    );
    
    // Initialize State
    goal_active_ = false;
    current_octree_ = nullptr;

    RCLCPP_INFO(get_logger(), "Local Planner Node initialized");
  }

private:
  // ---------------------------------------------------------------------------
  // Member Variables
  // ---------------------------------------------------------------------------
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr frontier_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr frontier_candidates_publisher_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // Data Storage
  std::shared_ptr<octomap::OcTree> current_octree_;
  geometry_msgs::msg::Point last_drone_position_;
  tf2::Quaternion last_drone_orientation_;
  
  // Moving average of drone movement direction
  std::vector<tf2::Vector3> movement_history_; 
  
  // Store frontier candidates for global planner
  std::vector<octomap::point3d> last_frontier_candidates_;


  // State Machine for Goal Persistence
  bool goal_active_;
  geometry_msgs::msg::Point current_goal_;

  // Configuration
  const float exploration_radius_scale = 4.0;       // Radius around frontier to calculate gain
  const float occupied_threshold_ = 0.5;            // Occupancy probability threshold
  const float search_radius_ = 80.0;                // Search radius
  const float exploration_weight_ = 2.0;            // Weight for exploration gain
  const float direction_weight_ = 1.0;              // Weight for direction gain
  const float x_limit_position_ = -370.0;           // Cave boundary limit
  const size_t movement_history_size_ = 10;         // Number of movement samples for moving average
  const float frontier_safety_radius_ = 1.0;        // Clearance radius around frontier - reduced from 2.0 to be less restrictive
  
  // Stability Parameters
  const double goal_reached_tolerance_ = 20.0;      // Meters. Drone is "close enough" to switch goals
  
  // ---------------------------------------------------------------------------
  // Callbacks
  // ---------------------------------------------------------------------------

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Calculate movement vector from last position
    geometry_msgs::msg::Point new_pos = msg->pose.pose.position;
    tf2::Vector3 movement(
      new_pos.x - last_drone_position_.x,
      new_pos.y - last_drone_position_.y,
      new_pos.z - last_drone_position_.z
    );
    
    // Add to movement history if significant movement occurred
    if (movement.length() > 0.1) {  // At least 10cm movement
      movement_history_.push_back(movement);
      
      // Keep only the last N samples
      if (movement_history_.size() > movement_history_size_) {
        movement_history_.erase(movement_history_.begin());
      }
    }
    
    last_drone_position_ = new_pos;
    last_drone_orientation_ = tf2::Quaternion(
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w);
  }

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Simply convert and store the tree.
    // We do NOT plan here. This prevents "twitching".
    octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
    if (tree) {
        // dynamic_cast to OcTree. If it fails, tree remains null/old one stays.
        octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);
        if (octree) {
            current_octree_ = std::shared_ptr<octomap::OcTree>(octree);
        } else {
            RCLCPP_ERROR(get_logger(), "Octomap message was not an OcTree");
            delete tree;
        }
    }
  }

  // ---------------------------------------------------------------------------
  // Planning Loop (The Core Logic)
  // ---------------------------------------------------------------------------

  void planning_loop()
  {
    if (!current_octree_) {
        RCLCPP_DEBUG(get_logger(), "Waiting for Octomap...");
        return;
    }

    bool need_new_goal = false;

    // 1. Check existing goal status
    if (goal_active_) {
        double dist = std::sqrt(
            std::pow(last_drone_position_.x - current_goal_.x, 2) +
            std::pow(last_drone_position_.y - current_goal_.y, 2) +
            std::pow(last_drone_position_.z - current_goal_.z, 2)
        );

        // Condition A: Reached?
        if (dist < goal_reached_tolerance_) {
            RCLCPP_INFO(get_logger(), "Goal reached (dist: %.2fm). Replanning.", dist);
            need_new_goal = true;
        }
        // Condition B: Blocked?
        else {
            octomap::point3d goal_pt(current_goal_.x, current_goal_.y, current_goal_.z);
            if (!is_point_visible(current_octree_.get(), goal_pt)) {
                 RCLCPP_WARN(get_logger(), "Current goal path obstructed. Replanning.");
                 need_new_goal = true;
            }
        }
    } else {
        need_new_goal = true;
    }

    // 2. Select Goal (if needed)
    if (need_new_goal) {
        geometry_msgs::msg::Point new_goal;
        float best_gain = find_best_frontier(new_goal);

        if (best_gain > 0.0) {
            current_goal_ = new_goal;
            goal_active_ = true;
            RCLCPP_INFO(get_logger(), "New Frontier Selected: [%.1f, %.1f, %.1f, Gain: %.2f]", 
                        new_goal.x, new_goal.y, new_goal.z, best_gain);
        } else {
            RCLCPP_WARN(get_logger(), "No valid frontiers found. Holding position.");
            goal_active_ = false;
        }
    }

    // 3. Publish Command
    // We publish even if the goal hasn't changed to keep the controller active.
    if (goal_active_) {
        publish_trajectory(current_goal_);
        publish_marker(current_goal_);
    }
    
    // 4. Publish frontier candidates for global planner
    publish_frontier_candidates();
  }

  // ---------------------------------------------------------------------------
  // Algorithm Implementations
  // ---------------------------------------------------------------------------

  float find_best_frontier(geometry_msgs::msg::Point& result_point)
  {
    // 1. Identify frontier points
    std::vector<octomap::point3d> frontier_points = get_frontier_points(current_octree_.get());
    
    // If no frontiers found or the frontier is the same 10 times in a row, check if we can backtrack
    // ToDo: Communication with global planner for backtracking
    if (frontier_points.empty()) {
      RCLCPP_WARN(get_logger(), "No frontier points found locally. Global planner will handle backtracking.");
    }
    RCLCPP_DEBUG(get_logger(), "Found %lu frontier points", frontier_points.size());
    
    // 2. Filter frontiers (Cave constraints)
    std::vector<octomap::point3d> cave_frontiers;
    for (const auto& point : frontier_points) {
      if (point.x() < x_limit_position_) {
        cave_frontiers.push_back(point);
      }
    }

    // Use cave frontiers if available
    const std::vector<octomap::point3d>& candidates = cave_frontiers.empty() ? frontier_points : cave_frontiers;

    // Store candidates for global planner
    last_frontier_candidates_ = candidates;

    // 3. Downsampling optimization
    std::vector<octomap::point3d> points_to_evaluate;
    if (candidates.size() > 50) {
      size_t stride = candidates.size() / 50;
      for (size_t i = 0; i < candidates.size(); i += stride) {
        points_to_evaluate.push_back(candidates[i]);
      }
    } else {
      points_to_evaluate = candidates;
    }

    // 4. Score points
    if (points_to_evaluate.empty()) {
      RCLCPP_WARN(get_logger(), "No points to evaluate after filtering");
      result_point.x = 0;
      result_point.y = 0;
      result_point.z = 0;
      return -1.0;
    }
    
    octomap::point3d best_point = points_to_evaluate[0];
    float best_gain = -1.0;
    
    for (const auto& point : points_to_evaluate) {
      double point_size = current_octree_->getNodeSize(current_octree_->getTreeDepth());
      float exploration_gain = calculate_exploration_gain(current_octree_.get(), point, exploration_radius_scale * point_size);
      float direction_gain = calculate_direction_gain(point);
      
      float total_gain = exploration_weight_ * exploration_gain + direction_weight_ * direction_gain;
      
      // Debug logging for direction gain analysis
      if (total_gain > best_gain * 0.9) {  // Log top candidates
        RCLCPP_INFO(get_logger(), 
          "Frontier: pos(%.1f,%.1f,%.1f) exp_gain=%.3f dir_gain=%.3f total=%.3f (hist_size=%lu)",
          point.x(), point.y(), point.z(), exploration_gain, direction_gain, total_gain, movement_history_.size());
      }
      
      if (total_gain > best_gain) {
        best_gain = total_gain;
        best_point = point;
      }
    }

    result_point.x = best_point.x();
    result_point.y = best_point.y();
    result_point.z = best_point.z();

    return best_gain;
  }

  std::vector<octomap::point3d> get_frontier_points(octomap::OcTree* octree)
  {
    std::vector<octomap::point3d> frontiers;
    octomap::point3d drone_position(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    
    // Iterate through ALL octree nodes (all depths), not just leaves
    // This ensures we find frontiers at multiple resolutions
    for (auto it = octree->begin(octree->getTreeDepth()); it != octree->end(); ++it) {
      // Optimization: Bounding box check first
      if (it.getCoordinate().distance(drone_position) > search_radius_) {
        continue;
      }

      // Only consider free cells
      if (it->getOccupancy() < occupied_threshold_) {
        double size = octree->getNodeSize(it.getDepth());
        if (size < 4.0) {
          continue; // Skip large nodes for frontier detection
        }
        octomap::point3d node_coordinate = it.getCoordinate();
        
        // Check 6 direct neighbors
        double directions[6][3] = {
          {size, 0, 0}, {-size, 0, 0},
          {0, size, 0}, {0, -size, 0},
          {0, 0, size}, {0, 0, -size}
        };
        
        bool is_frontier = false;
        bool is_safe = true;
        int unknown_neighbors = 0;
        int occupied_neighbors = 0;

        for (auto& dir : directions) {
          octomap::point3d neighbor_point = 
            node_coordinate + octomap::point3d(dir[0], dir[1], dir[2]);
          octomap::OcTreeNode* neighbor_node = octree->search(neighbor_point);
          
          // If neighbor doesn't exist in tree, it's unknown → this is a frontier
          if (!neighbor_node) {
            is_frontier = true;
            unknown_neighbors++;
          } else if (neighbor_node->getOccupancy() > occupied_threshold_) {
             // Neighbor is an obstacle
             occupied_neighbors++;
             // Don't immediately fail - frontiers can touch walls
          }
        }
        
        // Accept frontiers with at least 1 unknown neighbor (more permissive)
        // This allows finding frontiers in more situations
        if (is_frontier && unknown_neighbors >= 1) {
          is_safe = true;
        } else {
          is_safe = false;
        }
        
        if (is_frontier && is_safe) {
          // Check visibility from drone to prevent selecting unreachable points
          if (is_point_visible(octree, node_coordinate)) {
             // Check that no obstacles are too close to this frontier
             if (is_frontier_safe_clearance(octree, node_coordinate)) {
               frontiers.push_back(node_coordinate);
             }
          }
        }
      }
    }
    return frontiers;
  }

  bool is_point_visible(octomap::OcTree* octree, const octomap::point3d& target)
  {
    octomap::point3d drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    
    octomap::point3d direction = target - drone_pos;
    double length = direction.norm();

    if (length < 0.1) return true;

    octomap::point3d end_point;
    // castRay: true = ignore unknown (treat as free). Returns true if obstacle HIT.
    bool hit_obstacle = octree->castRay(drone_pos, direction, end_point, true, length);
    
    return !hit_obstacle;
  }

  bool is_frontier_safe_clearance(octomap::OcTree* octree, const octomap::point3d& frontier_point)
  {
    // Check that no occupied voxels exist too close to frontier
    // Use a smaller checking radius and fewer samples to be more permissive
    float check_step = 0.5f;  // Check every 0.5 meter
    int occupied_count = 0;
    int total_checks = 0;
    
    for (float dx = -frontier_safety_radius_; dx <= frontier_safety_radius_; dx += check_step) {
      for (float dy = -frontier_safety_radius_; dy <= frontier_safety_radius_; dy += check_step) {
        for (float dz = -frontier_safety_radius_; dz <= frontier_safety_radius_; dz += check_step) {
          octomap::point3d check_point = frontier_point + octomap::point3d(dx, dy, dz);
          octomap::OcTreeNode* node = octree->search(check_point);
          total_checks++;
          
          if (node && octree->isNodeOccupied(node)) {
            occupied_count++;
          }
        }
      }
    }
    
    // Allow frontier if less than 30% of the surrounding space is occupied
    // This is more permissive than requiring ZERO obstacles
    float occupancy_ratio = static_cast<float>(occupied_count) / static_cast<float>(total_checks);
    return occupancy_ratio < 0.3f;  // Allow some obstacles nearby
  }

  float calculate_exploration_gain(octomap::OcTree* octree, const octomap::point3d& point, float radius)
  {
    int unknown_voxels = 0;
    int occupied_voxels = 0;
    int total_voxels = 0;
    float voxel_size = octree->getResolution();
    float step_size = (voxel_size > 0.4f) ? voxel_size : 0.4f; // Optimization: Larger steps
    
    // 1. Calculate Standard Exploration Gain
    // Count unknown voxels (no node found = unobserved space) and occupied voxels
    for (float dx = -radius; dx <= radius; dx += step_size) {
      for (float dy = -radius; dy <= radius; dy += step_size) {
        for (float dz = -radius; dz <= radius; dz += step_size) {
          octomap::point3d search_point = point + octomap::point3d(dx, dy, dz);
          total_voxels++;
          
          octomap::OcTreeNode* node = octree->search(search_point);
          // If no node found at this point, it's unknown/unobserved
          if (!node) {
            unknown_voxels++;
          } else if (octree->isNodeOccupied(node)) {
            occupied_voxels++;
          }
        }
      }
    }

    if (total_voxels == 0) return 0.0f;
    
    float volume_gain = static_cast<float>(unknown_voxels) / static_cast<float>(total_voxels);
    
    // 2. Calculate wall proximity penalty
    // Frontiers surrounded by many obstacles should be penalized
    // This discourages selecting frontiers that lead into tight spaces
    float occupancy_ratio = static_cast<float>(occupied_voxels) / static_cast<float>(total_voxels);
    
    // Wall penalty: if >40% of radius is occupied, reduce gain
    float wall_penalty = 1.0f;
    if (occupancy_ratio > 0.4f) {
      // Penalty increases with occupancy: at 40% occupied = 1.0x gain, at 70% = 0.3x gain
      wall_penalty = std::max(0.3f, 1.0f - (occupancy_ratio - 0.4f) * 2.33f);
    }
    
    // Clamp to [0, 1]
    volume_gain = std::min(1.0f, volume_gain * wall_penalty);

    return volume_gain;
  }

  /**
   * Calculate direction gain based on drone movement history (moving average)
   * 
   * Biases frontier selection toward where the drone has been heading.
   * Returns [0, 1] where 1.0 = perfectly aligned, 0.0 = opposite direction
   */
  float calculate_direction_gain(const octomap::point3d& frontier_point)
  {
    // Use average movement direction if available
    tf2::Vector3 reference_direction(0, 0, 0);
    bool has_valid_movement = false;
    
    if (movement_history_.size() >= 3) {
      // Calculate average movement direction from history (only if we have substantial history)
      for (const auto& movement : movement_history_) {
        reference_direction += movement;
      }
      reference_direction /= movement_history_.size();
      
      if (reference_direction.length() > 0.1f) {
        reference_direction.normalize();
        has_valid_movement = true;
      }
    }
    
    // If hovering or insufficient movement, return neutral gain (don't bias)
    if (!has_valid_movement) {
      return 0.5f;  // Neutral gain - let exploration gain dominate
    }

    // Calculate frontier direction from drone
    tf2::Vector3 drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    tf2::Vector3 frontier_vec(frontier_point.x() - drone_pos.x(),
                              frontier_point.y() - drone_pos.y(),
                              frontier_point.z() - drone_pos.z());
    
    if (frontier_vec.length() == 0) return 0.5f;
    frontier_vec.normalize();

    // Compute dot product: 1 = same direction, -1 = opposite, 0 = perpendicular
    double dot = reference_direction.dot(frontier_vec);
    dot = std::clamp(dot, -1.0, 1.0);
    
    // Map [-1, 1] to [0.2, 1.0]: opposite gets 0.2 (still possible), aligned gets 1.0
    // This allows exploration in ANY direction while slightly preferring forward motion
    float gain = static_cast<float>(0.2 + (dot + 1.0) * 0.4);
    
    return gain;
  }

  // ---------------------------------------------------------------------------
  // Publishing Helpers
  // ---------------------------------------------------------------------------

  void publish_trajectory(const geometry_msgs::msg::Point& target)
  {
    trajectory_msgs::msg::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header.stamp = this->get_clock()->now();
    trajectory_msg.header.frame_id = "world"; // Ensure "map" or "world" is correct for your setup
    trajectory_msg.joint_names.push_back("base_link");

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
    geometry_msgs::msg::Transform transform;
    transform.translation.x = target.x;
    transform.translation.y = target.y;
    transform.translation.z = target.z;
    
    // Calculate desired yaw based on direction to frontier
    double dx = target.x - last_drone_position_.x;
    double dy = target.y - last_drone_position_.y;
    double yaw = atan2(dy, dx);

    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);

    transform.rotation.x = q.x();
    transform.rotation.y = q.y();
    transform.rotation.z = q.z();
    transform.rotation.w = q.w(); 

    point.transforms.push_back(transform);

    // Populate velocities and accelerations (Zero for waypoint mode)
    geometry_msgs::msg::Twist velocity;
    geometry_msgs::msg::Twist acceleration;
    point.velocities.push_back(velocity);
    point.accelerations.push_back(acceleration);

    trajectory_msg.points.push_back(point);
    
    trajectory_publisher_->publish(trajectory_msg);
    
    // Also publish the point for debug/legacy
    frontier_publisher_->publish(target);
  }

  void publish_marker(const geometry_msgs::msg::Point& target)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "frontier_goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position = target;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 2.0;
    marker.scale.y = 2.0;
    marker.scale.z = 2.0;
    marker.color.a = 0.8;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker_publisher_->publish(marker);
  }

  void publish_frontier_candidates()
  {
    if (last_frontier_candidates_.empty()) {
      return;
    }
    
    nav_msgs::msg::Path candidates_path;
    candidates_path.header.frame_id = "world";
    candidates_path.header.stamp = this->get_clock()->now();
    
    // Convert all frontier candidates to path poses
    for (const auto& candidate : last_frontier_candidates_) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = this->get_clock()->now();
      pose.pose.position.x = candidate.x();
      pose.pose.position.y = candidate.y();
      pose.pose.position.z = candidate.z();
      pose.pose.orientation.w = 1.0;
      
      candidates_path.poses.push_back(pose);
    }
    
    frontier_candidates_publisher_->publish(candidates_path);
  }
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LocalPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}