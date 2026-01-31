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
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

/**
 * LocalPlannerNode - Frontier-Based Exploration Local Planner with Global Backtracking
 * 
 * This ROS 2 node implements a local planner that:
 * 1. Subscribes to Octomap updates from the mapping system
 * 2. Identifies frontier cells (boundaries between explored and unexplored space)
 * 3. Calculates exploration gain for each frontier
 * 4. Publishes the frontier with highest gain for the trajectory planner to execute
 * 5. Tracks flight history and backtracks to explore other regions when no local frontiers found
 * 
 * ROS Interface (Subscribers):
 * - /octomap_full (octomap_msgs/Octomap) - 3D occupancy grid from Octomap
 * - /current_state (nav_msgs/Odometry) - Drone pose and orientation
 * 
 * ROS Interface (Publishers):
 * - /frontier_goal (geometry_msgs/Point) - Best frontier point for exploration
 * - /backtrack_path (nav_msgs/Path) - Path to backtrack through flown points to nearest unexplored frontier
 * - /frontier_goal_marker (visualization_msgs/Marker) - Visualization marker for frontier
 * - /command/trajectory (trajectory_msgs/MultiDOFJointTrajectory) - Trajectory command
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
    
    // ----- Subscribers -----
    // Subscribe to Octomap updates and bind it to octomap callback (does frontier evaluation)
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full",
      10,
      std::bind(&LocalPlannerNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to odometry for drone pose and orientation
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state", 10,
      std::bind(&LocalPlannerNode::odom_callback, this, std::placeholders::_1)
    );
    
    last_frontier_found_time_ = this->now();
    
    RCLCPP_INFO(get_logger(), "Local Planner Node initialized");
    RCLCPP_INFO(get_logger(), "Subscribing to /octomap_full");
    RCLCPP_INFO(get_logger(), "Publishing frontier goals to /frontier_goal");
    RCLCPP_INFO(get_logger(), "Global Planner handles backtracking when no local frontiers found");
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr frontier_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

  // Frontier info structure
  struct FrontierInfo {
    octomap::point3d position;
    float exploration_gain;
    bool visited;
    int id;
    
    FrontierInfo(octomap::point3d pos, float gain, int frontier_id)
      : position(pos), exploration_gain(gain), visited(false), id(frontier_id) {}
  };

  // Configuration
  const float exploration_radius_ = 10.0;    // Radius around frontier to calculate gain
  const float occupied_threshold_ = 0.5;    // Occupancy probability threshold
  const float search_radius_ = 50.0;        // Optimization: Search radius increased for larger caves (20m diam)
  const float frontier_proximity_threshold_ = 2.0;  // Distance to consider frontiers as same
  const size_t movement_history_size_ = 10;  // Number of movement samples for moving average

  // Store latest drone pose and orientation
  geometry_msgs::msg::Point last_drone_position_;
  tf2::Quaternion last_drone_orientation_;
  rclcpp::Time last_frontier_found_time_;
  
  // Moving average of drone movement direction
  std::vector<tf2::Vector3> movement_history_;  // History of drone movement vectors

  /**
   * Callback for Octomap subscription
   * 
   * Called whenever a new Octomap message is received.
   * Identifies frontiers, calculates gains, and publishes best frontier.
   */
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Convert ROS message to C++ octree object
    octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
    octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);
    
    // RCLCPP_WARN(get_logger(), "Octomap_callback triggered");

    // Check for valid octree
    if (!octree) {
      RCLCPP_WARN(get_logger(), "Failed to cast Octomap message to OcTree");
      delete tree;
      return;
    }
    
    // 1. Identify frontier points
    std::vector<octomap::point3d> frontier_points = get_frontier_points(octree);
    
    if (frontier_points.empty()) {
      RCLCPP_WARN(get_logger(), "No frontier points found locally. Global planner will handle backtracking.");
      delete tree;
      return;
    }    RCLCPP_DEBUG(get_logger(), "Found %lu frontier points", frontier_points.size());
    
    // 2. Find frontier with highest exploration gain
    
    // Filter frontiers to keep only those within the cave (x < -350)
    std::vector<octomap::point3d> cave_frontiers;
    for (const auto& point : frontier_points) {
      if (point.x() < -370.0) {
        cave_frontiers.push_back(point);
      }
    }

    // Use cave frontiers if available to enforce staying inside
    const std::vector<octomap::point3d>& candidates = cave_frontiers.empty() ? frontier_points : cave_frontiers;

    // Optimization: Downsample frontiers if there are too many to process in real-time
    std::vector<octomap::point3d> points_to_evaluate;
    if (candidates.size() > 50) {
      // Pick ~50 points distributed across the vector
      size_t stride = candidates.size() / 50;
      for (size_t i = 0; i < candidates.size(); i += stride) {
        points_to_evaluate.push_back(candidates[i]);
      }
    } else {
      points_to_evaluate = candidates;
    }

    octomap::point3d best_point = points_to_evaluate[0];
    float best_gain = 0.0;
    
    for (const auto& point : points_to_evaluate) {
      float exploration_gain = calculate_exploration_gain(octree, point, exploration_radius_);
      float direction_gain = calculate_direction_gain(point);
      
      // Combine gains (with equal weighting for now)
      float total_gain = 2*exploration_gain + direction_gain;
      
      if (total_gain > best_gain) {
        best_gain = total_gain;
        best_point = point;
      }
    }
    
    // 3. Publish best frontier
    geometry_msgs::msg::Point goal_msg;
    goal_msg.x = best_point.x();
    goal_msg.y = best_point.y();
    goal_msg.z = best_point.z();
    
    frontier_publisher_->publish(goal_msg);

    // Publish trajectory
    trajectory_msgs::msg::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header = msg->header;
    trajectory_msg.joint_names.push_back("base_link");

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
    geometry_msgs::msg::Transform transform;
    transform.translation.x = best_point.x();
    transform.translation.y = best_point.y();
    transform.translation.z = best_point.z();
    
    // Calculate desired yaw based on direction to frontier
    double dx = best_point.x() - last_drone_position_.x;
    double dy = best_point.y() - last_drone_position_.y;
    double yaw = atan2(dy, dx);

    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);

    transform.rotation.x = q.x();
    transform.rotation.y = q.y();
    transform.rotation.z = q.z();
    transform.rotation.w = q.w(); 

    point.transforms.push_back(transform);

    // Populate velocities and accelerations (required by controller)
    geometry_msgs::msg::Twist velocity;
    velocity.linear.x = 0.0;
    velocity.linear.y = 0.0;
    velocity.linear.z = 0.0;
    velocity.angular.x = 0.0;
    velocity.angular.y = 0.0;
    velocity.angular.z = 0.0;
    point.velocities.push_back(velocity);

    geometry_msgs::msg::Twist acceleration;
    acceleration.linear.x = 0.0;
    acceleration.linear.y = 0.0;
    acceleration.linear.z = 0.0;
    acceleration.angular.x = 0.0;
    acceleration.angular.y = 0.0;
    acceleration.angular.z = 0.0;
    point.accelerations.push_back(acceleration);

    trajectory_msg.points.push_back(point);
    
    trajectory_publisher_->publish(trajectory_msg);

    // Publish marker for visualization in RViz
    visualization_msgs::msg::Marker marker;
    marker.header = msg->header; // Use the same frame as the octomap
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "frontier_goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position = goal_msg;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 5.0;
    marker.scale.y = 5.0;
    marker.scale.z = 5.0;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker_publisher_->publish(marker);
    
    RCLCPP_INFO(get_logger(),
                "Published frontier goal: (%.2f, %.2f, %.2f) with gain %.2f",
                goal_msg.x, goal_msg.y, goal_msg.z, best_gain);
    
    // Reset timer for local frontier discovery
    last_frontier_found_time_ = this->now();
    
    // Cleanup
    delete tree;
  }

  /**
   * Add frontier to discovered frontiers list, avoiding duplicates
   */

  /**
   * Check if a ray from the drone to the target point is unblocked.
   * Returns true if visible (no obstacles in path), false otherwise.
   */
  bool is_point_visible(octomap::OcTree* octree, const octomap::point3d& target)
  {
    octomap::point3d drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    
    // Direction vector from drone to target
    octomap::point3d direction = target - drone_pos;
    double length = direction.norm();

    // If extremely close, consider it visible
    if (length < 0.1) return true;

    octomap::point3d end_point;
    
    // castRay returns true if an obstacle is hit within the specified length.
    // Since we want center points, any hit before the target means it is occluded.
    // 'true' in 4th arg means ignore unknown cells (treat as free), only stop on occupied.
    bool hit_obstacle = octree->castRay(drone_pos, direction, end_point, true, length);
    
    if (hit_obstacle) {
      return false; // Blocked by wall
    }
    
    return true; // Clear path
  }

  std::vector<octomap::point3d> get_frontier_points(octomap::OcTree* octree)
  {
    std::vector<octomap::point3d> frontiers;
    octomap::point3d drone_position(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    
    for (auto leaf = octree->begin_leafs(); leaf != octree->end_leafs(); ++leaf) {
      // Optimization: Bounding box check first (fastest) - skip far away points
      if (leaf.getCoordinate().distance(drone_position) > search_radius_) {
        continue;
      }

      // Only consider free cells
      if (leaf->getOccupancy() < occupied_threshold_) {
        double size = octree->getNodeSize(leaf.getDepth());
        octomap::point3d leaf_coordinate = leaf.getCoordinate();
        
        // Check 6 direct neighbors
        double directions[6][3] = {
          {size, 0, 0}, {-size, 0, 0},
          {0, size, 0}, {0, -size, 0},
          {0, 0, size}, {0, 0, -size}
        };
        
        bool is_frontier = false;
        bool is_safe = true;

        for (auto& dir : directions) {
          octomap::point3d neighbor_point = 
            leaf_coordinate + octomap::point3d(dir[0], dir[1], dir[2]);
          octomap::OcTreeNode* neighbor_node = octree->search(neighbor_point);
          
          // If neighbor doesn't exist in tree, it's unknown → this is a frontier
          if (!neighbor_node) {
            is_frontier = true;
          } else if (neighbor_node->getOccupancy() > occupied_threshold_) {
             // Neighbor is an obstacle -> unsafe
             is_safe = false;
             break;
          }
        }
        
        if (is_frontier && is_safe) {
          // Only add frontier if it is visible from the drone
          if (is_point_visible(octree, leaf_coordinate)) {
             frontiers.push_back(leaf_coordinate);
          }
        }
      }
    }
    return frontiers;
  }

  /**
   * Calculate exploration gain around a point
   * 
   * Exploration gain is normalized to [0, 1] representing the proportion of
   * unknown voxels within the search radius. A voxel is unknown if it doesn't 
   * exist in the Octomap (has never been observed).
   * Higher gain means more unexplored space nearby.
   */
  float calculate_exploration_gain(octomap::OcTree* octree,
                                   const octomap::point3d& point,
                                   float radius)
  {
    int unknown_voxels = 0;
    int total_voxels = 0;
    float voxel_size = octree->getResolution();
    float step_size = (voxel_size > 0.2f) ? voxel_size : 0.2f;
    
    // 1. Calculate Standard Exploration Gain (Volume of Unknowns)
    for (float dx = -radius; dx <= radius; dx += step_size) {
      for (float dy = -radius; dy <= radius; dy += step_size) {
        for (float dz = -radius; dz <= radius; dz += step_size) {
          octomap::point3d search_point = point + octomap::point3d(dx, dy, dz);
          total_voxels++;
          if (!octree->search(search_point)) {
            unknown_voxels++;
          }
        }
      }
    }

    if (total_voxels == 0) return 0.0;
    float volume_gain = static_cast<float>(unknown_voxels) / static_cast<float>(total_voxels);

    // 2. Calculate "Center of Cave" Penalty (Proximity to Obstacles)
    // We check a small safety sphere (e.g., 1.0m) around the frontier.
    // If we find occupied cells, we apply a penalty.
    float safety_radius = 1.0; 
    bool near_obstacle = false;
    
    // Quick check: Look at 6 directions around the point
    double check_dirs[6][3] = {
        {safety_radius, 0, 0}, {-safety_radius, 0, 0},
        {0, safety_radius, 0}, {0, -safety_radius, 0},
        {0, 0, safety_radius}, {0, 0, -safety_radius}
    };

    for (auto& dir : check_dirs) {
        octomap::point3d check_p = point + octomap::point3d(dir[0], dir[1], dir[2]);
        octomap::OcTreeNode* node = octree->search(check_p);
        if (node && octree->isNodeOccupied(node)) {
            near_obstacle = true;
            break; 
        }
    }

    // Apply Penalty
    // If near a wall, reduce gain by 40%. This biases the planner 
    // to choose points with similar unknown volume but simpler geometry (center).
    if (near_obstacle) {
        return volume_gain * 0.6; 
    }

    return volume_gain;
  }

  /**
   * Calculate direction gain based on drone movement history (moving average)
   * 
   * Instead of just using camera heading, this calculates gain based on the
   * drone's overall movement direction over the last N samples (moving average).
   * This biases frontier selection toward where the drone has been heading,
   * making exploration more coherent and continuous.
   * 
   * Returns [0, 1] where 1.0 = perfectly aligned with movement direction, 0.0 = opposite
   */
  float calculate_direction_gain(const octomap::point3d& frontier_point)
  {
    // If we don't have enough movement history, fall back to camera heading
    if (movement_history_.size() < 2) {
      tf2::Vector3 cam_forward_body(1, 0, 0);
      tf2::Vector3 cam_forward_world = tf2::quatRotate(last_drone_orientation_, cam_forward_body);
      
      tf2::Vector3 drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
      tf2::Vector3 frontier_vec(frontier_point.x() - drone_pos.x(),
                                frontier_point.y() - drone_pos.y(),
                                frontier_point.z() - drone_pos.z());
      if (frontier_vec.length() == 0) return 0.0f;
      frontier_vec.normalize();
      cam_forward_world.normalize();
      
      double dot = cam_forward_world.dot(frontier_vec);
      dot = std::clamp(dot, -1.0, 1.0);
      float gain = static_cast<float>(cos(acos(dot)));
      return (gain + 1.0f) / 2.0f;
    }
    
    // Calculate average movement direction from history
    tf2::Vector3 avg_movement(0, 0, 0);
    for (const auto& movement : movement_history_) {
      avg_movement += movement;
    }
    avg_movement /= movement_history_.size();
    
    if (avg_movement.length() < 0.01f) {
      // Drone hasn't moved much, return neutral gain
      return 0.5f;
    }
    avg_movement.normalize();

    // Vector from drone to frontier in world frame
    tf2::Vector3 drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    tf2::Vector3 frontier_vec(frontier_point.x() - drone_pos.x(),
                              frontier_point.y() - drone_pos.y(),
                              frontier_point.z() - drone_pos.z());
    if (frontier_vec.length() == 0) return 0.0f;
    frontier_vec.normalize();

    // Compute angle between average movement direction and frontier direction
    double dot = avg_movement.dot(frontier_vec);
    dot = std::clamp(dot, -1.0, 1.0);
    double angle = acos(dot);

    // Normalize: 1.0 if perfectly aligned, 0.0 if opposite
    float gain = static_cast<float>(cos(angle));
    gain = (gain + 1.0f) / 2.0f;
    return gain;
  }


  
  // Store latest drone pose and orientation, and track movement history
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
};



int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<LocalPlannerNode>();
  
  RCLCPP_INFO(node->get_logger(), "Local Planner Node is running...");
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}
