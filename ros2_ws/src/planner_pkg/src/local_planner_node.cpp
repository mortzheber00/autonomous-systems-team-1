#include <rclcpp/rclcpp.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>

/**
 * LocalPlannerNode - Frontier-Based Exploration Local Planner
 * 
 * This ROS 2 node implements a local planner that:
 * 1. Subscribes to Octomap updates from the mapping system
 * 2. Identifies frontier cells (boundaries between explored and unexplored space)
 * 3. Calculates exploration gain for each frontier
 * 4. Publishes the frontier with highest gain for the trajectory planner to execute
 * 
 * ROS Interface (Subscribers):
 * - /octomap_full (octomap_msgs/Octomap) - 3D occupancy grid from Octomap
 * 
 * ROS Interface (Publishers):
 * - /frontier_goal (geometry_msgs/Point) - Best frontier point for exploration
 */
class LocalPlannerNode : public rclcpp::Node
{
public:
  LocalPlannerNode() : Node("local_planner_node")
  {
    // Create publisher for frontier goals
    frontier_publisher_ = create_publisher<geometry_msgs::msg::Point>("/frontier_goal", 10);
    marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>("/frontier_goal_marker", 10);
    
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
    
    RCLCPP_INFO(get_logger(), "Local Planner Node initialized");
    RCLCPP_INFO(get_logger(), "Subscribing to /octomap_full");
    RCLCPP_INFO(get_logger(), "Publishing frontier goals to /frontier_goal");
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr frontier_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

  // Configuration
  const float exploration_radius_ = 1.0;    // Radius around frontier to calculate gain
  const float occupied_threshold_ = 0.5;    // Occupancy probability threshold

  // Store latest drone pose and orientation
  geometry_msgs::msg::Point last_drone_position_;
  tf2::Quaternion last_drone_orientation_;

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
    
    // Check for valid octree
    if (!octree) {
      RCLCPP_WARN(get_logger(), "Failed to cast Octomap message to OcTree");
      delete tree;
      return;
    }
    
    // 1. Identify frontier points
    std::vector<octomap::point3d> frontier_points = get_frontier_points(octree);
    
    if (frontier_points.empty()) {
      RCLCPP_DEBUG(get_logger(), "No frontier points found in Octomap");
      delete tree;
      return;
    }
    
    RCLCPP_DEBUG(get_logger(), "Found %lu frontier points", frontier_points.size());
    
    // 2. Find frontier with highest exploration gain
    // Optimization: Downsample frontiers if there are too many to process in real-time
    std::vector<octomap::point3d> points_to_evaluate;
    if (frontier_points.size() > 50) {
      // Pick ~50 points distributed across the vector
      size_t stride = frontier_points.size() / 50;
      for (size_t i = 0; i < frontier_points.size(); i += stride) {
        points_to_evaluate.push_back(frontier_points[i]);
      }
    } else {
      points_to_evaluate = frontier_points;
    }

    octomap::point3d best_point = points_to_evaluate[0];
    float best_gain = 0.0;
    
    for (const auto& point : points_to_evaluate) {
      float exploration_gain = calculate_exploration_gain(octree, point, exploration_radius_);
      float direction_gain = calculate_direction_gain(point);
      
      // Combine gains (with equal weighting for now)
      float total_gain = exploration_gain + direction_gain;
      
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

    // Publish marker for visualization in RViz
    visualization_msgs::msg::Marker marker;
    marker.header = msg->header; // Use the same frame as the octomap
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "frontier_goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = goal_msg.x;
    marker.pose.position.y = goal_msg.y;
    marker.pose.position.z = goal_msg.z;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 10.0;
    marker.scale.y = 10.0;
    marker.scale.z = 10.0;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker_publisher_->publish(marker);
    
    RCLCPP_INFO(get_logger(),
                "Published frontier goal: (%.2f, %.2f, %.2f) with gain %.2f",
                goal_msg.x, goal_msg.y, goal_msg.z, best_gain);
    
    // Cleanup
    delete tree;
  }

  /**
   * Identify all frontier cells in the Octomap
   * 
   * A frontier cell is a free voxel adjacent to at least one unknown voxel.
   * These represent the boundary between explored and unexplored space.
   */
  std::vector<octomap::point3d> get_frontier_points(octomap::OcTree* octree)
  {
    std::vector<octomap::point3d> frontiers;
    
    for (auto leaf = octree->begin_leafs(); leaf != octree->end_leafs(); ++leaf) {
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
        for (auto& dir : directions) {
          octomap::point3d neighbor_point = 
            leaf_coordinate + octomap::point3d(dir[0], dir[1], dir[2]);
          octomap::OcTreeNode* neighbor_node = octree->search(neighbor_point);
          
          // If neighbor doesn't exist in tree, it's unknown → this is a frontier
          if (!neighbor_node) {
            is_frontier = true;
            break;
          }
        }
        
        if (is_frontier) {
          frontiers.push_back(leaf_coordinate);
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
    // Count unknown and total voxels in a grid around the frontier point
    int unknown_voxels = 0;
    int total_voxels = 0;
    float voxel_size = octree->getResolution();
    // Optimization: Use a coarser step size for gain calculation (min 0.2m) to avoid exponential cost
    float step_size = (voxel_size > 0.2f) ? voxel_size : 0.2f;
    
    // Create a search grid around the point
    for (float dx = -radius; dx <= radius; dx += step_size) {
      for (float dy = -radius; dy <= radius; dy += step_size) {
        for (float dz = -radius; dz <= radius; dz += step_size) {
          octomap::point3d search_point = point + octomap::point3d(dx, dy, dz);
          total_voxels++;
          
          // If the voxel doesn't exist in the octree, it's unknown
          if (!octree->search(search_point)) {
            unknown_voxels++;
          }
        }
      }
    }
    // Return normalized gain [0, 1]
    if (total_voxels == 0) {
      return 0.0;
    }
    return static_cast<float>(unknown_voxels) / static_cast<float>(total_voxels);
  }

  /**
   * Calculate direction gain based on drone heading
   * 
   * Exploration gain is higher when the frontier is aligned with the drone's
   * current heading direction. This function calculates the angle between the
   * camera view direction and the frontier direction, normalizing the gain to [0,1].
   * (1 = perfectly aligned, 0 = perpendicular or opposite)
   */
  float calculate_direction_gain(const octomap::point3d& frontier_point)
  {
    // Camera forward vector in drone body frame (assume x-axis)
    tf2::Vector3 cam_forward_body(1, 0, 0);
    // Rotate to world frame using last known orientation
    tf2::Vector3 cam_forward_world = tf2::quatRotate(last_drone_orientation_, cam_forward_body);

    // Vector from drone to frontier in world frame
    tf2::Vector3 drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    tf2::Vector3 frontier_vec(frontier_point.x() - drone_pos.x(),
                              frontier_point.y() - drone_pos.y(),
                              frontier_point.z() - drone_pos.z());
    if (frontier_vec.length() == 0) return 0.0f;
    frontier_vec.normalize();
    cam_forward_world.normalize();

    // Compute angle between camera view and frontier direction
    double dot = cam_forward_world.dot(frontier_vec);
    dot = std::clamp(dot, -1.0, 1.0); // Clamp for safety
    double angle = acos(dot); // [0, pi]

    // Normalize: 1.0 if perfectly aligned, 0.0 if opposite
    float gain = static_cast<float>(cos(angle)); // [-1,1]
    gain = (gain + 1.0f) / 2.0f; // [0,1]
    return gain;
  }


  
  // Store latest drone pose and orientation
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_drone_position_ = msg->pose.pose.position;
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
