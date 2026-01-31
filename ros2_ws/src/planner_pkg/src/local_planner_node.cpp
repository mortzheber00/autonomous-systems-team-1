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
#include <visualization_msgs/msg/marker.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

/**
 * LocalPlannerNode - Frontier-Based Exploration Local Planner
 * * IMPROVEMENTS:
 * 1. Decoupled Planning from Mapping: Uses a 2Hz timer for planning instead of reacting to every map update.
 * 2. Goal Commitment: Stick to a goal until reached (tolerance) or blocked. Prevents twitching.
 * 3. Constant Trajectory Publishing: Keeps the controller happy even when not changing goals.
 */
class LocalPlannerNode : public rclcpp::Node
{
public:
  LocalPlannerNode() : Node("local_planner_node")
  {
    // Publishers
    frontier_publisher_ = create_publisher<geometry_msgs::msg::Point>("/frontier_goal", 10);
    marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>("/frontier_goal_marker", 10);
    trajectory_publisher_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/command/trajectory", 10);
    
    // Subscriber: Octomap (Only updates internal map structure)
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

    // Timer: Run planning loop at 2 Hz (500ms)
    // This stabilizes the behavior by not replanning on every single map voxel change.
    planning_timer_ = create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&LocalPlannerNode::planning_loop, this)
    );
    
    // Initialize State
    goal_active_ = false;
    current_octree_ = nullptr;

    RCLCPP_INFO(get_logger(), "Local Planner Node initialized with Stabilized Loop (2Hz)");
  }

private:
  // ---------------------------------------------------------------------------
  // Member Variables
  // ---------------------------------------------------------------------------
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr frontier_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // Data Storage
  std::shared_ptr<octomap::OcTree> current_octree_;
  geometry_msgs::msg::Point last_drone_position_;
  tf2::Quaternion last_drone_orientation_;

  // State Machine for Goal Persistence
  bool goal_active_;
  geometry_msgs::msg::Point current_goal_;

  // Configuration
  const float exploration_radius_ = 5.0;    // Radius around frontier to calculate gain
  const float occupied_threshold_ = 0.5;    // Occupancy probability threshold
  const float search_radius_ = 40.0;        // Search radius
  const float exploration_weight_ = 2.0;    // Weight for exploration gain
  const float direction_weight_ = 1.0;      // Weight for direction gain
  const float x_limit_position_ = -370.0;   // Cave boundary limit
  
  // Stability Parameters
  const double goal_reached_tolerance_ = 8.0; // Meters. Drone is "close enough" to switch goals
  
  // ---------------------------------------------------------------------------
  // Callbacks
  // ---------------------------------------------------------------------------

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_drone_position_ = msg->pose.pose.position;
    last_drone_orientation_ = tf2::Quaternion(
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w);
  }

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Simply convert and store the tree.
    // We do NOT plan here. This prevents the "twitching".
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
  }

  // ---------------------------------------------------------------------------
  // Algorithm Implementations
  // ---------------------------------------------------------------------------

  float find_best_frontier(geometry_msgs::msg::Point& result_point)
  {
    // 1. Identify frontier points
    std::vector<octomap::point3d> frontier_points = get_frontier_points(current_octree_.get());
    
    if (frontier_points.empty()) return 0.0;
    
    // 2. Filter frontiers (Cave constraints)
    std::vector<octomap::point3d> cave_frontiers;
    for (const auto& point : frontier_points) {
      if (point.x() < x_limit_position_) {
        cave_frontiers.push_back(point);
      }
    }

    // Use cave frontiers if available
    const std::vector<octomap::point3d>& candidates = cave_frontiers.empty() ? frontier_points : cave_frontiers;

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
    octomap::point3d best_point = points_to_evaluate[0];
    float best_gain = -1.0;
    
    for (const auto& point : points_to_evaluate) {
      float exploration_gain = calculate_exploration_gain(current_octree_.get(), point, exploration_radius_);
      float direction_gain = calculate_direction_gain(point);
      
      float total_gain = exploration_weight_ * exploration_gain + direction_weight_ * direction_gain;
      
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
    
    for (auto leaf = octree->begin_leafs(); leaf != octree->end_leafs(); ++leaf) {
      // Optimization: Bounding box check first
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
          // Check visibility from drone to prevent selecting unreachable points
          if (is_point_visible(octree, leaf_coordinate)) {
             frontiers.push_back(leaf_coordinate);
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

  float calculate_exploration_gain(octomap::OcTree* octree, const octomap::point3d& point, float radius)
  {
    int unknown_voxels = 0;
    int total_voxels = 0;
    float voxel_size = octree->getResolution();
    float step_size = (voxel_size > 0.4f) ? voxel_size : 0.4f; // Optimization: Larger steps
    
    // 1. Calculate Standard Exploration Gain
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

    // 2. Safety Penalty
    float safety_radius = 4.0; 
    bool near_obstacle = false;
    
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

    if (near_obstacle) {
        return volume_gain * 0.6; 
    }

    return volume_gain;
  }

  float calculate_direction_gain(const octomap::point3d& frontier_point)
  {
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
    
    // Remap cosine [-1, 1] to [0, 1]
    return static_cast<float>((dot + 1.0) / 2.0);
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
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LocalPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}