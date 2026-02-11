/**
 * @file global_planner_node.cpp written by Phil Chudalla
 * @brief Global Planner Node for Autonomous Drone Exploration
 * This ROS 2 node saves a sparse set of frontiers from the local planner and continously updates their exploration gain.
 * It also keeps track of the flown points fo backtracking.
 * When the local planner cant find a valid frontier, the global planner selects the best unvisited frontier from its list and publishes it as a new goal.
 * The drone can then backtrack through the flown points list to reach the new goal.
 * When the drone is in range of the new goal frontier, the local planner takes over again for fine exploration.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <octomap/octomap.h>
#include <cmath>
#include <octomap_msgs/conversions.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>

/**
 * FrontierInfo - Structure to store frontier data
 */
struct FrontierInfo {
  geometry_msgs::msg::Point position;
  float exploration_gain;
  bool visited;
  int id;
  
  FrontierInfo(geometry_msgs::msg::Point pos, float gain, int frontier_id)
    : position(pos), exploration_gain(gain), visited(false), id(frontier_id) {}
};

/**
 * GlobalPlannerNode - Global Planner for Frontier-Based Exploration
 * 
 * This ROS 2 node implements a global planner that:
 * 1. Maintains a sparse set of frontiers from the local planner
 * 2. Continuously updates exploration gains for each frontier
 * 3. Tracks the drone's flight path for backtracking
 * 4. Selects the best frontier when local planner fails
 * 5. Commands backtracking when needed
 * 
 * ROS Interface (Subscribers):
 * - /frontier_goals (geometry_msgs/PoseArray) - Local planner's frontiers
 * - /true_pose (nav_msgs/Odometry) - Current drone position and heading
 * - /path (nav_msgs/Path) - Path from RRT planner
 * 
 * ROS Interface (Publishers):
 * - /global_goal (geometry_msgs/Point) - Best unvisited frontier when local planner fails
 * - /backtrack_path (nav_msgs/Path) - Path for drone to backtrack through flown points
 */
class GlobalPlannerNode : public rclcpp::Node
{
public:
  GlobalPlannerNode() : Node("global_planner_node")
  {
    // ----- Publishers -----
    // Create publisher for backtrack path (to guide drone back through flown points)
    backtrack_path_publisher_ = create_publisher<nav_msgs::msg::Path>("/backtrack_path", 10);
    
    // Create publisher for backtrack goals as PoseArray (flown points + goal frontier → sent to RRT)
    backtrack_goals_publisher_ = create_publisher<geometry_msgs::msg::PoseArray>("/backtrack_goals", 10);
    
    // Create publisher for frontier visualization
    frontier_markers_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("/global_frontier_markers", 10);
    
    // Create publisher for flown points visualization
    flown_points_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("/flown_points_markers", 10);
    
    // ----- Subscribers -----
    // Subscribe to local planner active status
    local_planner_active_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/local_planner_active", 10,
      std::bind(&GlobalPlannerNode::local_planner_active_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to frontier goals from local planner to build frontier list
    frontier_subscription_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/frontier_goals", 10,
      std::bind(&GlobalPlannerNode::frontier_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to Octomap for exploration gain calculation
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full", 10,
      std::bind(&GlobalPlannerNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to drone odometry for position tracking and flight path recording
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state_est", 10,
      std::bind(&GlobalPlannerNode::odom_callback, this, std::placeholders::_1)
    );
    
    // Create timer for periodic updates and global planning
    planning_timer_ = create_wall_timer(
      std::chrono::milliseconds(1000),
      std::bind(&GlobalPlannerNode::planning_loop, this)
    );
    
    // Create timer for visualization updates
    viz_timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&GlobalPlannerNode::publish_frontier_markers, this)
    );
    
    RCLCPP_INFO(get_logger(), "Global Planner Node initialized");
    RCLCPP_INFO(get_logger(), "Maintaining sparse frontier set with periodic pruning");
    RCLCPP_INFO(get_logger(), "Frontiers added: 5 -> Prune low-gain frontiers and re-evaluate gains");
  }

private:
  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr backtrack_path_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr backtrack_goals_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_markers_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr flown_points_publisher_;
  
  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr frontier_subscription_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr local_planner_active_subscription_;
  
  // Timers
  rclcpp::TimerBase::SharedPtr planning_timer_;
  rclcpp::TimerBase::SharedPtr viz_timer_;
  
  // Octomap storage
  std::shared_ptr<octomap::OcTree> octree_;
  
  // State tracking
  bool local_planner_active_ = true;    // Whether local planner is finding frontiers
  bool backtrack_published_ = false;    // Whether we already published a backtrack goal
  
  // Frontier management
  std::vector<FrontierInfo> frontier_list_;
  int frontier_counter_ = 0;
  int frontier_added_since_prune_ = 0;  // Counter for frontier additions
  const float frontier_proximity_threshold_ = 5.0;  // Distance to consider frontiers as same (reduced from 10m)
  const int prune_interval_ = 5;                    // Prune every 5 new frontiers
  const float min_gain_threshold_ = 0.15;            // Minimum gain to keep frontier (reduced from 2.0 - gains are [0,1])
  const float exploration_radius_ = 5.0;            // Radius for exploration gain calculation
  
  // Global frontier discovery parameters
  const float global_frontier_spacing_ = 20.0;     // Minimum distance between global frontiers (sparse)
  const float volumetric_radius_ = 8.0;            // Radius for volumetric gain calculation
  const float min_global_gain_ = 0.4;              // Minimum volumetric gain for global frontiers
  int frontier_discovery_counter_ = 0;              // Timer for periodic frontier discovery
  
  // Flight tracking
  std::vector<geometry_msgs::msg::Point> flown_points_;
  geometry_msgs::msg::Point current_position_;
  
  // Configuration
  const size_t max_frontiers_ = 100;  // Maximum number of frontiers to maintain

  /**
   * Callback for frontier goals from local planner
   * 
   * Receives frontiers from local planner and adds them to sparse frontier set.
   * Gains will be re-evaluated based on Octomap exploration gains.
   * Every 5 frontiers added, re-evaluates and prunes low-gain frontiers.
   */
  void frontier_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    // Process each pose in the array
    for (const auto& pose : msg->poses) {
      geometry_msgs::msg::Point point = pose.position;
      
      // Check if frontier is close to existing frontier
      auto existing = find_nearby_frontier(point);
      
      if (existing != frontier_list_.end()) {
        // Frontier already exists nearby - will be re-evaluated with Octomap
        // Skip duplicate (no logging to reduce spam)
      } else {
        // Add new frontier if under limit
        if (frontier_list_.size() < max_frontiers_) {
          frontier_list_.emplace_back(point, 0.5, frontier_counter_++);
          frontier_added_since_prune_++;
          RCLCPP_INFO(get_logger(), "[FRONTIER] Added new frontier (ID: %d) at (%.2f, %.2f, %.2f). Total: %lu",
                      frontier_counter_ - 1, point.x, point.y, point.z, frontier_list_.size());
          
          // After every 5 new frontiers, re-evaluate gains and prune
          if (frontier_added_since_prune_ >= prune_interval_) {
            RCLCPP_INFO(get_logger(), "[PRUNE] Reached interval (%d frontiers added). Re-evaluating gains and pruning...",
                        frontier_added_since_prune_);
            prune_low_gain_frontiers();
            frontier_added_since_prune_ = 0;
          }
        } else {
          RCLCPP_WARN(get_logger(), "Frontier list at maximum size (%lu), skipping new frontier at (%.2f, %.2f, %.2f)", 
                      frontier_list_.size(), point.x, point.y, point.z);
        }
      }
    }
  }

  /**
   * Callback for Octomap updates
   * 
   * Re-evaluates exploration gains for all frontiers based on current Octomap state.
   */
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Convert message to OcTree
    auto octree = std::make_shared<octomap::OcTree>(0.1);
    octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
    if (tree) {
      octree = std::make_shared<octomap::OcTree>(*dynamic_cast<octomap::OcTree*>(tree));
      delete tree;
    } else {
      RCLCPP_WARN(get_logger(), "Failed to convert Octomap message");
      return;
    }
    
    octree_ = octree;
    
    // Re-evaluate gains for all frontiers
    for (auto& frontier : frontier_list_) {
      if (!frontier.visited) {
        frontier.exploration_gain = calculate_exploration_gain(frontier.position);
      }
    }
    
    RCLCPP_DEBUG(get_logger(), "Re-evaluated exploration gains for %lu frontiers", frontier_list_.size());
  }

  /**
   * Callback for drone odometry
   * 
   * Tracks current drone position and adds to flight path.
   */
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    current_position_ = msg->pose.pose.position;
    
    // Add to flown points if far enough from last point
    if (flown_points_.empty() ||
        distance(flown_points_.back(), current_position_) > 10.0) {
      flown_points_.push_back(current_position_);
      publish_flown_points_markers();
    }
  }

  /**
   * Callback for local planner active status
   * 
   * When local planner goes inactive, triggers backtracking.
   * When local planner becomes active again, resets backtrack state.
   */
  void local_planner_active_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    bool was_active = local_planner_active_;
    local_planner_active_ = msg->data;
    
    if (was_active && !local_planner_active_) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] Local planner went INACTIVE - triggering backtrack planning");
      backtrack_published_ = false;  // Allow new backtrack to be published
    } else if (!was_active && local_planner_active_) {
      RCLCPP_INFO(get_logger(), "[GLOBAL] Local planner is ACTIVE again - stopping backtrack");
      backtrack_published_ = false;
    }
  }

  /**
   * Main planning loop - called periodically
   * 
   * When local planner fails to find frontiers (signal via log or subscription),
   * this selects the best high-gain frontier and publishes backtrack path.
   * The drone follows the flown path in reverse to reach the next frontier.
   */
  void planning_loop()
  {
    // Periodically discover new frontiers (every 3 seconds)
    frontier_discovery_counter_++;
    if (frontier_discovery_counter_ >= 3 && octree_) {
      discover_global_frontiers();
      frontier_discovery_counter_ = 0;
    }
    
    // Always log current frontier count for visibility
    size_t total_frontiers = frontier_list_.size();
    size_t unvisited_frontiers = std::count_if(frontier_list_.begin(), frontier_list_.end(),
                                               [](const FrontierInfo& f) { return !f.visited; });
    
    // Find highest gain among unvisited frontiers
    float highest_gain = 0.0f;
    for (const auto& frontier : frontier_list_) {
      if (!frontier.visited && frontier.exploration_gain > highest_gain) {
        highest_gain = frontier.exploration_gain;
      }
    }
    
    RCLCPP_INFO(get_logger(), "[GLOBAL] Frontier list: %zu total (%zu unvisited), highest gain: %.3f", 
                total_frontiers, unvisited_frontiers, highest_gain);
    
    // Only act when local planner is inactive (can't find good frontiers)
    if (local_planner_active_) {
      return;  // Local planner is handling exploration
    }
    
    // Don't re-publish if we already sent a backtrack goal
    if (backtrack_published_) {
      return;
    }
    
    // Need frontiers and flown points to backtrack
    if (frontier_list_.empty()) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] No frontiers in list for backtracking");
      return;
    }
    if (flown_points_.empty()) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] No flown points recorded for backtracking");
      return;
    }
    
    FrontierInfo* best_frontier = select_best_frontier();
    if (!best_frontier) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] No unvisited frontiers available");
      return;
    }
    
    RCLCPP_INFO(get_logger(), 
                "[GLOBAL] Backtracking to frontier (ID: %d, Gain: %.3f) at (%.2f, %.2f, %.2f)",
                best_frontier->id, best_frontier->exploration_gain,
                best_frontier->position.x, best_frontier->position.y, best_frontier->position.z);
    
    // Build backtrack waypoint path and publish to RRT
    publish_backtrack_goals(best_frontier);
    backtrack_published_ = true;
    
    // Mark frontier as visited so we don't try it again
    best_frontier->visited = true;
  }

  /**
   * Find frontier within proximity threshold
   * 
   * @param point - Point to search around
   * @return Iterator to nearby frontier, or end() if none found
   */
  std::vector<FrontierInfo>::iterator find_nearby_frontier(const geometry_msgs::msg::Point& point)
  {
    for (auto it = frontier_list_.begin(); it != frontier_list_.end(); ++it) {
      if (distance(it->position, point) < frontier_proximity_threshold_) {
        return it;
      }
    }
    return frontier_list_.end();
  }

  /**
   * Select best unvisited frontier
   * 
   * Finds the frontier with highest exploration gain that hasn't been visited.
   * Prioritizes frontiers based on exploration gain.
   * 
   * @return Best frontier or nullptr if none available
   */
  FrontierInfo* select_best_frontier()
  {
    FrontierInfo* best = nullptr;
    float best_score = -1.0;
    
    for (auto& frontier : frontier_list_) {
      if (frontier.visited) continue;  // Skip visited frontiers
      
      // Score based primarily on exploration gain
      float score = frontier.exploration_gain;
      
      if (score > best_score) {
        best_score = score;
        best = &frontier;
      }
    }
    
    return best;
  }

  /**
   * Publish backtrack goals as PoseArray to RRT
   * 
   * Builds an ordered list of waypoints:
   * 1. Find the flown point nearest to the target frontier
   * 2. Build path from current position through flown points to that nearest point
   * 3. Append the target frontier as the final goal
   * 
   * The RRT will plan segment-by-segment through these waypoints.
   * 
   * @param target_frontier - The frontier to navigate towards
   */
  void publish_backtrack_goals(FrontierInfo* target_frontier)
  {
    if (!target_frontier || flown_points_.empty()) {
      return;
    }
    
    // Step 1: Find flown point nearest to target frontier
    size_t nearest_idx = 0;
    float nearest_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < flown_points_.size(); ++i) {
      float dist = distance(flown_points_[i], target_frontier->position);
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest_idx = i;
      }
    }
    
    // Step 2: Find flown point nearest to current drone position
    size_t current_idx = 0;
    float current_nearest_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < flown_points_.size(); ++i) {
      float dist = distance(flown_points_[i], current_position_);
      if (dist < current_nearest_dist) {
        current_nearest_dist = dist;
        current_idx = i;
      }
    }
    
    // Step 3: Build waypoint list from current_idx to nearest_idx through flown points
    geometry_msgs::msg::PoseArray backtrack_goals;
    backtrack_goals.header.stamp = get_clock()->now();
    backtrack_goals.header.frame_id = "world";
    
    // Add flown points from current position toward the frontier
    // Determine direction (forward or backward through flown_points_)
    if (current_idx <= nearest_idx) {
      // Forward through flown points
      for (size_t i = current_idx; i <= nearest_idx; ++i) {
        geometry_msgs::msg::Pose pose;
        pose.position = flown_points_[i];
        pose.orientation.w = 1.0;
        backtrack_goals.poses.push_back(pose);
      }
    } else {
      // Backward through flown points (backtracking)
      for (size_t i = current_idx; i >= nearest_idx; --i) {
        geometry_msgs::msg::Pose pose;
        pose.position = flown_points_[i];
        pose.orientation.w = 1.0;
        backtrack_goals.poses.push_back(pose);
        if (i == 0) break;  // Prevent underflow
      }
    }
    
    // Step 4: Append the target frontier as the final goal
    geometry_msgs::msg::Pose frontier_pose;
    frontier_pose.position = target_frontier->position;
    frontier_pose.orientation.w = 1.0;
    backtrack_goals.poses.push_back(frontier_pose);
    
    backtrack_goals_publisher_->publish(backtrack_goals);
    
    RCLCPP_INFO(get_logger(), 
               "[GLOBAL] Published backtrack goals: %zu waypoints (flown_pts[%zu→%zu] + frontier ID:%d)",
               backtrack_goals.poses.size(), current_idx, nearest_idx, target_frontier->id);
    
    // Also publish as nav_msgs/Path for visualization
    nav_msgs::msg::Path backtrack_path;
    backtrack_path.header = backtrack_goals.header;
    for (const auto& pose : backtrack_goals.poses) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = backtrack_goals.header;
      ps.pose = pose;
      backtrack_path.poses.push_back(ps);
    }
    backtrack_path_publisher_->publish(backtrack_path);
  }

  /**
   * Calculate Euclidean distance between two points
   */
  float distance(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2)
  {
    return std::sqrt(
      std::pow(p1.x - p2.x, 2) +
      std::pow(p1.y - p2.y, 2) +
      std::pow(p1.z - p2.z, 2)
    );
  }

  /**
   * Calculate exploration gain for a frontier position
   * 
   * Computes the ratio of unknown voxels to total voxels within exploration radius.
   * Returns normalized value in [0, 1].
   * 
   * @param position - Frontier position to evaluate
   * @return Normalized exploration gain [0, 1]
   */
  float calculate_exploration_gain(const geometry_msgs::msg::Point& position)
  {
    if (!octree_) {
      return 0.0f;  // No Octomap yet
    }
    
    int unknown_count = 0;
    int total_count = 0;
    float voxel_size = octree_->getResolution();
    
    // Sample points in exploration radius to check for unknown space
    // Use a grid-based sampling approach to properly check unexplored areas
    for (float x = position.x - exploration_radius_; x <= position.x + exploration_radius_; x += voxel_size) {
      for (float y = position.y - exploration_radius_; y <= position.y + exploration_radius_; y += voxel_size) {
        for (float z = position.z - exploration_radius_; z <= position.z + exploration_radius_; z += voxel_size) {
          
          // Check if point is within exploration radius
          float dist = sqrt(pow(x - position.x, 2) + pow(y - position.y, 2) + pow(z - position.z, 2));
          if (dist > exploration_radius_) continue;
          
          total_count++;
          
          // Check if this location is unknown/unexplored
          octomap::point3d query_point(x, y, z);
          octomap::OcTreeNode* node = octree_->search(query_point);
          
          if (!node) {
            // Node doesn't exist = unexplored/unknown space
            unknown_count++;
          } else {
            // Node exists but check if it's unknown (log odds ~ 0)
            if (abs(node->getLogOdds()) < 0.1) {  // Near-zero log odds = unknown
              unknown_count++;
            }
          }
        }
      }
    }
    
    // Calculate normalized gain
    if (total_count == 0) {
      return 0.0f;
    }
    
    float gain = static_cast<float>(unknown_count) / static_cast<float>(total_count);
    return std::min(1.0f, gain);  // Clamp to [0, 1]
  }

  /**
   * Prune low-gain frontiers from the frontier list
   * 
   * Removes frontiers with exploration gain below threshold, keeping only high-quality frontiers.
   * Skips visited frontiers to allow them to be explored later if needed.
   * Gain is based on current Octomap state (unknown voxels around frontier).
   */
  void prune_low_gain_frontiers()
  {
    int removed_count = 0;
    size_t initial_size = frontier_list_.size();
    
    // Remove frontiers with gain below threshold (only unvisited ones)
    frontier_list_.erase(
      std::remove_if(frontier_list_.begin(), frontier_list_.end(),
        [&](const FrontierInfo& frontier) {
          if (!frontier.visited && frontier.exploration_gain < min_gain_threshold_) {
            RCLCPP_INFO(get_logger(), 
                        "Pruning frontier ID %d with low gain %.2f at (%.2f, %.2f, %.2f)",
                        frontier.id, frontier.exploration_gain,
                        frontier.position.x, frontier.position.y, frontier.position.z);
            removed_count++;
            return true;
          }
          return false;
        }),
      frontier_list_.end()
    );
    
    if (removed_count > 0) {
      RCLCPP_INFO(get_logger(), 
                 "Frontier list pruning: Removed %d low-gain frontiers. "
                 "List size: %lu → %lu (Unvisited: %d)",
                 removed_count, initial_size, frontier_list_.size(),
                 count_unvisited_frontiers());
    } else {
      RCLCPP_INFO(get_logger(), 
                 "Frontier list pruning: No frontiers removed (all have gain >= %.2f)",
                 min_gain_threshold_);
    }
  }

  /**
   * Count unvisited frontiers in the list
   */
  int count_unvisited_frontiers() const
  {
    int count = 0;
    for (const auto& frontier : frontier_list_) {
      if (!frontier.visited) count++;
    }
    return count;
  }

  /**
   * Publish frontier markers for RViz visualization
   * 
   * Creates markers for:
   * - Unvisited frontiers (green spheres)
   * - Visited frontiers (gray spheres)
   * - Best frontier (large yellow sphere with outline)
   */
  void publish_frontier_markers()
  {
    visualization_msgs::msg::MarkerArray marker_array;
    
    // Clear previous markers
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);
    
    int marker_id = 0;
    FrontierInfo* best_frontier = select_best_frontier();
    
    for (const auto& frontier : frontier_list_) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = get_clock()->now();
      marker.ns = "frontiers";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      
      marker.pose.position.x = frontier.position.x;
      marker.pose.position.y = frontier.position.y;
      marker.pose.position.z = frontier.position.z;
      marker.pose.orientation.w = 1.0;
      
      // Standard marker size - 0.5m diameter spheres
      marker.scale.x = 0.5;
      marker.scale.y = 0.5;
      marker.scale.z = 0.5;
      
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(2));
      
      // Color based on status
      if (frontier.visited) {
        // Visited frontiers: gray
        marker.color.r = 0.5f;
        marker.color.g = 0.5f;
        marker.color.b = 0.5f;
        marker.color.a = 0.6f;
      } else if (best_frontier && frontier.id == best_frontier->id) {
        // Best frontier: large yellow with high visibility
        marker.color.r = 1.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
      } else {
        // Unvisited frontiers: green
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.8f;
      }
      
      marker_array.markers.push_back(marker);
    }
    
    // Add text marker showing total frontier count
    if (!frontier_list_.empty() && current_position_.x != 0 && current_position_.y != 0) {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = "world";
      text_marker.header.stamp = get_clock()->now();
      text_marker.ns = "frontier_stats";
      text_marker.id = 10000;  // Use high ID to avoid conflicts
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      
      text_marker.pose.position = current_position_;
      text_marker.pose.position.z += 3.0;  // Offset above drone
      text_marker.pose.orientation.w = 1.0;
      
      text_marker.scale.z = 1.5;  // Text height for better readability
      text_marker.color.r = 1.0f;
      text_marker.color.g = 1.0f;
      text_marker.color.b = 1.0f;
      text_marker.color.a = 1.0f;
      
      int visited_count = 0;
      for (const auto& f : frontier_list_) {
        if (f.visited) visited_count++;
      }
      
      char text_buffer[256];
      snprintf(text_buffer, sizeof(text_buffer),
               "Frontiers: %lu\nVisited: %d\nUnvisited: %d\nBest Gain: %.2f",
               frontier_list_.size(), visited_count, count_unvisited_frontiers(),
               best_frontier ? best_frontier->exploration_gain : 0.0f);
      text_marker.text = text_buffer;
      
      text_marker.lifetime = rclcpp::Duration(std::chrono::seconds(2));
      marker_array.markers.push_back(text_marker);
    }
    
    frontier_markers_publisher_->publish(marker_array);
    
    // Debug logging
    if (!frontier_list_.empty()) {
      RCLCPP_DEBUG(get_logger(), "Published %lu markers with frame_id='map' (Unvisited: %d, Visited: %d)",
                   frontier_list_.size(), count_unvisited_frontiers(),
                   static_cast<int>(frontier_list_.size()) - count_unvisited_frontiers());
    }
  }

  /**
   * Publish visualization markers for flown points
   */
  void publish_flown_points_markers()
  {
    visualization_msgs::msg::MarkerArray marker_array;
    
    for (size_t i = 0; i < flown_points_.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = get_clock()->now();
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = flown_points_[i];
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.5;
      marker.scale.y = 0.5;
      marker.scale.z = 0.5;
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;  // Green
      marker.color.b = 0.0f;
      marker.color.a = 1.0f;
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(0));  // Persistent
      
      marker_array.markers.push_back(marker);
    }
    
    flown_points_publisher_->publish(marker_array);
  }

  // ---------------------------------------------------------------------------
  // Global Frontier Discovery
  // ---------------------------------------------------------------------------

  /**
   * Discover global frontiers based on volumetric gain only
   * Creates a sparse set of high-value exploration targets
   */
  void discover_global_frontiers()
  {
    if (!octree_) return;
    
    RCLCPP_DEBUG(get_logger(), "[GLOBAL] Discovering frontiers from octree...");
    
    // Get all frontier points in the environment
    std::vector<octomap::point3d> candidate_frontiers = get_global_frontier_points();
    
    int added_count = 0;
    for (const auto& point : candidate_frontiers) {
      // Calculate pure volumetric gain (no direction bias)
      float volumetric_gain = calculate_volumetric_gain(point, volumetric_radius_);
      
      // Skip low-gain frontiers
      if (volumetric_gain < min_global_gain_) {
        continue;
      }
      
      // Check if too close to existing frontiers (maintain sparsity)
      geometry_msgs::msg::Point geo_point;
      geo_point.x = point.x();
      geo_point.y = point.y();
      geo_point.z = point.z();
      
      auto nearby = find_nearby_frontier(geo_point, global_frontier_spacing_);
      if (nearby != frontier_list_.end()) {
        continue; // Too close to existing frontier
      }
      
      // Add new global frontier
      if (frontier_list_.size() < max_frontiers_) {
        frontier_list_.emplace_back(geo_point, volumetric_gain, frontier_counter_++);
        added_count++;
        
        RCLCPP_INFO(get_logger(), "[GLOBAL] Discovered frontier (ID: %d) at (%.2f, %.2f, %.2f) gain=%.3f",
                    frontier_counter_ - 1, point.x(), point.y(), point.z(), volumetric_gain);
        
        if (added_count >= 3) break; // Limit discovery per cycle
      }
    }
    
    if (added_count > 0) {
      RCLCPP_INFO(get_logger(), "[GLOBAL] Added %d new frontiers, total: %zu", 
                  added_count, frontier_list_.size());
    }
  }

  /**
   * Find frontier points in the global octree
   * A frontier is at the boundary between free and unknown space
   */
  std::vector<octomap::point3d> get_global_frontier_points()
  {
    std::vector<octomap::point3d> frontiers;
    if (!octree_) return frontiers;
    
    double resolution = octree_->getResolution();
    
    // Traditional frontier detection: Free space adjacent to unknown
    for (auto it = octree_->begin_leafs(); it != octree_->end_leafs(); ++it) {
      if (octree_->isNodeOccupied(*it)) continue;  // Only free space
      
      octomap::point3d center = it.getCoordinate();
      
      // Check 26 neighbors (3x3x3 cube minus center) for unknown space
      bool is_frontier = false;
      for (int dx = -1; dx <= 1 && !is_frontier; dx++) {
        for (int dy = -1; dy <= 1 && !is_frontier; dy++) {
          for (int dz = -1; dz <= 1 && !is_frontier; dz++) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            
            octomap::point3d neighbor(
              center.x() + dx * resolution,
              center.y() + dy * resolution, 
              center.z() + dz * resolution
            );
            
            octomap::OcTreeNode* neighbor_node = octree_->search(neighbor);
            if (!neighbor_node) {
              // Unknown neighbor = potential frontier
              is_frontier = true;
            }
          }
        }
      }
      
      if (is_frontier) {
        frontiers.push_back(center);
      }
    }
    
    RCLCPP_DEBUG(get_logger(), "[GLOBAL] Found %zu traditional frontiers", frontiers.size());
    return frontiers;
  }

  /**
   * Calculate volumetric gain for a frontier point
   * Pure exploration potential based on unknown voxels nearby
   */
  float calculate_volumetric_gain(const octomap::point3d& point, float radius)
  {
    if (!octree_) return 0.0f;
    
    int unknown_voxels = 0;
    int total_voxels = 0;
    double resolution = octree_->getResolution();
    float step_size = std::max(resolution, 1.0); // Step by at least 1m for efficiency
    
    for (float dx = -radius; dx <= radius; dx += step_size) {
      for (float dy = -radius; dy <= radius; dy += step_size) {
        for (float dz = -radius; dz <= radius; dz += step_size) {
          // Skip if outside spherical radius
          float dist = sqrt(dx*dx + dy*dy + dz*dz);
          if (dist > radius) continue;
          
          octomap::point3d sample(point.x() + dx, point.y() + dy, point.z() + dz);
          octomap::OcTreeNode* node = octree_->search(sample);
          
          total_voxels++;
          if (!node) {
            // Unknown voxel
            unknown_voxels++;
          }
        }
      }
    }
    
    return total_voxels > 0 ? static_cast<float>(unknown_voxels) / total_voxels : 0.0f;
  }

  /**
   * Find nearby frontier with custom distance threshold
   */
  std::vector<FrontierInfo>::iterator find_nearby_frontier(const geometry_msgs::msg::Point& point, float threshold)
  {
    return std::find_if(frontier_list_.begin(), frontier_list_.end(),
      [&point, threshold, this](const FrontierInfo& frontier) {
        float dist = distance(frontier.position, point);
        return dist < threshold;
      });
  }
};


int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}