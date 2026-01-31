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
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include <sstream>

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
 * - /frontier_goal (geometry_msgs/Point) - Local planner's frontier
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
    
    // Create publisher for frontier visualization
    frontier_markers_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("/frontier_markers", 10);
    
    // ----- Subscribers -----
    // Subscribe to frontier goals from local planner to build frontier list
    frontier_subscription_ = create_subscription<geometry_msgs::msg::Point>(
      "/frontier_goal", 10,
      std::bind(&GlobalPlannerNode::frontier_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to Octomap for exploration gain calculation
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full", 10,
      std::bind(&GlobalPlannerNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to drone odometry for position tracking and flight path recording
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/true_pose", 10,
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
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_markers_publisher_;
  
  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr frontier_subscription_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  
  // Timers
  rclcpp::TimerBase::SharedPtr planning_timer_;
  rclcpp::TimerBase::SharedPtr viz_timer_;
  
  // Octomap storage
  std::shared_ptr<octomap::OcTree> octree_;
  
  // Frontier management
  std::vector<FrontierInfo> frontier_list_;
  int frontier_counter_ = 0;
  int frontier_added_since_prune_ = 0;  // Counter for frontier additions
  const float frontier_proximity_threshold_ = 5.0;  // Distance to consider frontiers as same (reduced from 10m)
  const int prune_interval_ = 5;                    // Prune every 5 new frontiers
  const float min_gain_threshold_ = 0.15;            // Minimum gain to keep frontier (reduced from 2.0 - gains are [0,1])
  const float exploration_radius_ = 5.0;            // Radius for exploration gain calculation
  
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
  void frontier_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    // Check if frontier is close to existing frontier
    auto existing = find_nearby_frontier(*msg);
    
    if (existing != frontier_list_.end()) {
      // Frontier already exists nearby - will be re-evaluated with Octomap
      RCLCPP_INFO(get_logger(), "Frontier already in list (ID: %d) at distance %.2f. Skipping duplicate.",
                   existing->id, distance(existing->position, *msg));
    } else {
      // Add new frontier if under limit
      if (frontier_list_.size() < max_frontiers_) {
        frontier_list_.emplace_back(*msg, 0.5, frontier_counter_++);
        frontier_added_since_prune_++;
        RCLCPP_INFO(get_logger(), "[FRONTIER] Added new frontier (ID: %d) at (%.2f, %.2f, %.2f). Total: %lu",
                    frontier_counter_ - 1, msg->x, msg->y, msg->z, frontier_list_.size());
        
        // After every 5 new frontiers, re-evaluate gains and prune
        if (frontier_added_since_prune_ >= prune_interval_) {
          RCLCPP_INFO(get_logger(), "[PRUNE] Reached interval (%d frontiers added). Re-evaluating gains and pruning...",
                      frontier_added_since_prune_);
          prune_low_gain_frontiers();
          frontier_added_since_prune_ = 0;
        }
      } else {
        RCLCPP_WARN(get_logger(), "Frontier list at maximum size (%lu), skipping new frontier at (%.2f, %.2f, %.2f)", 
                    frontier_list_.size(), msg->x, msg->y, msg->z);
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
        distance(flown_points_.back(), current_position_) > 0.5) {
      flown_points_.push_back(current_position_);
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
    // If we have frontiers and flown points, be ready to publish backtrack path
    if (!frontier_list_.empty() && !flown_points_.empty()) {
      FrontierInfo* best_frontier = select_best_frontier();
      
      if (best_frontier) {
        RCLCPP_INFO(get_logger(), 
                    "[BACKTRACK] Best frontier (ID: %d, Gain: %.3f) at (%.2f, %.2f, %.2f). "
                    "Current pos: (%.2f, %.2f, %.2f). Flown points: %lu.",
                    best_frontier->id, best_frontier->exploration_gain,
                    best_frontier->position.x, best_frontier->position.y, best_frontier->position.z,
                    current_position_.x, current_position_.y, current_position_.z,
                    flown_points_.size());
        
        // Publish backtrack path to guide drone through flown points to reach best frontier
        publish_backtrack_path(best_frontier);
      }
    } else if (frontier_list_.empty()) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] No frontiers in list yet");
    } else if (flown_points_.empty()) {
      RCLCPP_WARN(get_logger(), "[GLOBAL] No flown points recorded yet. Frontiers: %lu", frontier_list_.size());
    }
  }
    }
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
   * Publish backtrack path
   * 
   * Creates a path through flown points in reverse order for drone to backtrack to frontier.
   * The drone follows this path to reach the next high-gain frontier for exploration.
   * 
   * @param target_frontier - The frontier to backtrack towards
   */
  void publish_backtrack_path(FrontierInfo* target_frontier)
  {
    if (!target_frontier) {
      return;
    }
    
    nav_msgs::msg::Path backtrack_path;
    backtrack_path.header.frame_id = "map";
    backtrack_path.header.stamp = get_clock()->now();
    
    // Add flown points in reverse order (backtrack from current position back through path)
    for (auto it = flown_points_.rbegin(); it != flown_points_.rend(); ++it) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.header.stamp = get_clock()->now();
      pose.pose.position = *it;
      backtrack_path.poses.push_back(pose);
    }
    
    // Add the target frontier as final waypoint
    geometry_msgs::msg::PoseStamped final_pose;
    final_pose.header.frame_id = "map";
    final_pose.header.stamp = get_clock()->now();
    final_pose.pose.position = target_frontier->position;
    backtrack_path.poses.push_back(final_pose);
    
    backtrack_path_publisher_->publish(backtrack_path);
    RCLCPP_INFO(get_logger(), 
               "Published backtrack path: %lu flown points + frontier (ID: %d, Gain: %.2f) as target",
               flown_points_.size(), target_frontier->id, target_frontier->exploration_gain);
  }

  /**
   * Update frontier status
   * 
   * Marks frontier as visited when drone reaches it.
   * 
   * @param frontier - Frontier to mark as visited
   */
  void mark_frontier_visited(FrontierInfo& frontier)
  {
    frontier.visited = true;
    RCLCPP_INFO(get_logger(), "Marked frontier %d as visited", frontier.id);
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
    
    // Iterate through Octomap to count unknown voxels in radius
    for (auto it = octree_->begin(octree_->getTreeDepth()); 
         it != octree_->end(); ++it) {
      geometry_msgs::msg::Point voxel_pos;
      voxel_pos.x = it.getX();
      voxel_pos.y = it.getY();
      voxel_pos.z = it.getZ();
      
      float dist = distance(position, voxel_pos);
      if (dist <= exploration_radius_) {
        total_count++;
        
        // Check if voxel is unknown (not occupied, not free in observed space)
        if (it->getLogOdds() == 0) {  // Unknown voxels have log odds of 0
          unknown_count++;
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
      marker.header.frame_id = "map";
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
      text_marker.header.frame_id = "map";
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
};


int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}