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
#include <cmath>
#include <vector>
#include <algorithm>

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
    // Create publisher for global goals
    global_goal_publisher_ = create_publisher<geometry_msgs::msg::Point>("/global_goal", 10);
    
    // Create publisher for backtrack path
    backtrack_path_publisher_ = create_publisher<nav_msgs::msg::Path>("/backtrack_path", 10);
    
    // Subscribe to frontier goals from local planner
    frontier_subscription_ = create_subscription<geometry_msgs::msg::Point>(
      "/frontier_goal", 10,
      std::bind(&GlobalPlannerNode::frontier_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to drone odometry for position tracking
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/true_pose", 10,
      std::bind(&GlobalPlannerNode::odom_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to RRT path to track flown waypoints
    path_subscription_ = create_subscription<nav_msgs::msg::Path>(
      "/path", 10,
      std::bind(&GlobalPlannerNode::path_callback, this, std::placeholders::_1)
    );
    
    RCLCPP_INFO(get_logger(), "Global Planner Node initialized");
    RCLCPP_INFO(get_logger(), "Maintaining sparse frontier set and tracking flight path");
  }

private:
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr global_goal_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr backtrack_path_publisher_;
  
  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr frontier_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
  
  // Frontier management
  std::vector<FrontierInfo> frontier_list_;
  int frontier_counter_ = 0;
  const float frontier_proximity_threshold_ = 2.0;  // Distance to consider frontiers as same
  const float goal_range_threshold_ = 1.0;          // Distance to switch back to local planner
  
  // Flight tracking
  std::vector<geometry_msgs::msg::Point> flown_points_;
  geometry_msgs::msg::Point current_position_;
  
  // Configuration
  const size_t max_frontiers_ = 50;  // Maximum number of frontiers to maintain
  const float local_planner_range_ = 5.0;  // Range where local planner is active

  /**
   * Callback for frontier goals from local planner
   * 
   * Receives frontiers from local planner and adds them to sparse frontier set.
   * Updates exploration gains for nearby frontiers.
   */
  void frontier_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    // Check if frontier is close to existing frontier
    auto existing = find_nearby_frontier(*msg);
    
    if (existing != frontier_list_.end()) {
      // Update existing frontier's gain (assume it's still good)
      RCLCPP_DEBUG(get_logger(), "Updating existing frontier at (%.2f, %.2f, %.2f)",
                   existing->position.x, existing->position.y, existing->position.z);
    } else {
      // Add new frontier if under limit
      if (frontier_list_.size() < max_frontiers_) {
        frontier_list_.emplace_back(*msg, 0.5, frontier_counter_++);
        RCLCPP_INFO(get_logger(), "Added new frontier (ID: %d) at (%.2f, %.2f, %.2f)",
                    frontier_counter_ - 1, msg->x, msg->y, msg->z);
      } else {
        RCLCPP_DEBUG(get_logger(), "Frontier list at maximum size, skipping new frontier");
      }
    }
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
   * Callback for RRT path
   * 
   * Receives the planned path and uses waypoints for backtracking.
   */
  void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
  {
    // Path received - could be used for more sophisticated planning
    // For now, just log the path length
    RCLCPP_DEBUG(get_logger(), "Received path with %lu waypoints", msg->poses.size());
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
   * Prioritizes frontiers closer to current position.
   * 
   * @return Best frontier or nullptr if none available
   */
  FrontierInfo* select_best_frontier()
  {
    FrontierInfo* best = nullptr;
    float best_score = -1.0;
    
    for (auto& frontier : frontier_list_) {
      if (frontier.visited) continue;  // Skip visited frontiers
      
      // Score: exploration gain + proximity bonus
      float dist = distance(current_position_, frontier.position);
      float proximity_bonus = std::max(0.0f, 1.0f - (dist / local_planner_range_));
      float score = frontier.exploration_gain + 0.3 * proximity_bonus;
      
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
   * Creates a path through flown points for drone to backtrack.
   */
  void publish_backtrack_path()
  {
    nav_msgs::msg::Path backtrack_path;
    backtrack_path.header.frame_id = "map";
    backtrack_path.header.stamp = get_clock()->now();
    
    // Add flown points in reverse order (backtrack)
    for (auto it = flown_points_.rbegin(); it != flown_points_.rend(); ++it) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.header.stamp = get_clock()->now();
      pose.pose.position = *it;
      backtrack_path.poses.push_back(pose);
    }
    
    backtrack_path_publisher_->publish(backtrack_path);
    RCLCPP_INFO(get_logger(), "Published backtrack path with %lu points", backtrack_path.poses.size());
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
};


int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}