/**
 * @file local_planner_node.cpp written by Phil Chudalla
 * @brief Simplified Frontier-Based Exploration Local Planner
 * This ROS 2 node implements a simplified local planner that:
 * 1. Subscribes to Octomap updates from the mapping system
 * 2. Identifies frontier cells (boundaries between explored and unexplored space)
 * 3. Calculates exploration gain based on:
 *    - Volumetric exploration gain (unknown voxels around frontier)
 *    - Direction alignment with drone's current exploration direction
 * 4. Publishes top 5 frontiers to the RRT planner
 * 
 * Safety checks and path validation are handled by the RRT planner.
 */

#include <rclcpp/rclcpp.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

/**
 * LocalPlannerNode - Simplified Frontier-Based Exploration Local Planner
 * 
 * Simplified architecture:
 * - Finds frontiers and calculates basic gains
 * - Publishes best frontier to RRT planner
 * - RRT handles safety checks and trajectory generation
 */
class LocalPlannerNode : public rclcpp::Node
{
public:
  LocalPlannerNode() : Node("local_planner_node")
  {
    // --- Publishers -----
    // Create publisher for top 5 frontier goals (to RRT planner)
    frontier_publisher_ = create_publisher<geometry_msgs::msg::PoseArray>("/frontier_goals", 10);
    marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("/frontier_goals_markers", 10);
    
    // ----- Subscribers -----
    // Subscribe to Octomap updates
    octomap_subscription_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full",
      rclcpp::QoS(10).best_effort(),
      std::bind(&LocalPlannerNode::octomap_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to odometry for position and movement tracking
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state_est", 10,
      std::bind(&LocalPlannerNode::odom_callback, this, std::placeholders::_1)
    );

    // Timer: Run planning loop at 2 Hz (500ms)
    planning_timer_ = create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&LocalPlannerNode::planning_loop, this)
    );
    
    // Initialize State
    current_octree_ = nullptr;

    RCLCPP_INFO(get_logger(), "Simplified Local Planner Node initialized (2Hz planning)");
    RCLCPP_INFO(get_logger(), "Publishing top 5 frontiers to RRT planner for safety-aware path planning");
  }


private:
  // ---------------------------------------------------------------------------
  // Member Variables
  // ---------------------------------------------------------------------------
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr frontier_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // Data Storage
  std::shared_ptr<octomap::OcTree> current_octree_;
  geometry_msgs::msg::Point last_drone_position_;
  
  // Moving average of drone movement direction
  std::vector<tf2::Vector3> movement_history_; 

  // Configuration
  const float exploration_radius_ = 10.0;    // Radius around frontier to calculate gain
  const float occupied_threshold_ = 0.5;     // Occupancy probability threshold
  const float search_radius_ = 60.0;         // Search radius for frontiers
  const float x_limit_position_ = -370.0;    // Cave boundary limit
  const size_t movement_history_size_ = 10;  // Number of movement samples for moving average
  const float min_publish_gain_ = 1.3f;      // Minimum gain to publish a frontier (filters dead-end noise)
  
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
  }

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Convert and store the tree
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

  // ---------------------------------------------------------------------------
  // Planning Loop (The Core Logic)
  // ---------------------------------------------------------------------------

  void planning_loop()
  {
    if (!current_octree_) {
        RCLCPP_DEBUG(get_logger(), "Waiting for Octomap...");
        return;
    }

    // Find top 5 frontiers
    std::vector<geometry_msgs::msg::Point> top_frontiers = find_top_frontiers(5);

    if (!top_frontiers.empty()) {
        RCLCPP_INFO(get_logger(), "Publishing %zu frontiers to RRT", top_frontiers.size());
        
        // Publish frontiers as PoseArray
        geometry_msgs::msg::PoseArray frontier_array;
        frontier_array.header.stamp = this->get_clock()->now();
        frontier_array.header.frame_id = "world";
        
        for (const auto& frontier : top_frontiers) {
            geometry_msgs::msg::Pose pose;
            pose.position = frontier;
            pose.orientation.w = 1.0;
            frontier_array.poses.push_back(pose);
        }
        
        frontier_publisher_->publish(frontier_array);
        publish_markers(top_frontiers);
    }
  }

  // ---------------------------------------------------------------------------
  // Algorithm Implementations
  // ---------------------------------------------------------------------------

  std::vector<geometry_msgs::msg::Point> find_top_frontiers(size_t num_frontiers)
  {
    // 1. Identify frontier points
    std::vector<octomap::point3d> frontier_points = get_frontier_points(current_octree_.get());
    
    if (frontier_points.empty()) return {};
    
    // 2. Filter frontiers (Cave constraints)
    std::vector<octomap::point3d> cave_frontiers;
    for (const auto& point : frontier_points) {
      if (point.x() < x_limit_position_) {
        cave_frontiers.push_back(point);
      }
    }

    // Use cave frontiers if available, otherwise all frontiers
    const std::vector<octomap::point3d>& candidates = cave_frontiers.empty() ? frontier_points : cave_frontiers;

    // 3. Downsampling optimization (if too many candidates)
    std::vector<octomap::point3d> points_to_evaluate;
    if (candidates.size() > 50) {
      size_t stride = candidates.size() / 50;
      for (size_t i = 0; i < candidates.size(); i += stride) {
        points_to_evaluate.push_back(candidates[i]);
      }
    } else {
      points_to_evaluate = candidates;
    }

    // 4. Score all points and store in vector with gains
    struct ScoredFrontier {
      octomap::point3d point;
      float gain;
    };
    std::vector<ScoredFrontier> scored_frontiers;
    
    for (const auto& point : points_to_evaluate) {
      // Volumetric exploration gain
      float volume_gain = calculate_volumetric_gain(current_octree_.get(), point, exploration_radius_);
      
      // Direction alignment gain
      float direction_gain = calculate_direction_gain(point);
      
      // Combined gain
      float total_gain = volume_gain * direction_gain;
      
      scored_frontiers.push_back({point, total_gain});
    }

    // 5. Filter out low-gain frontiers (dead-end / already-explored noise)
    scored_frontiers.erase(
      std::remove_if(scored_frontiers.begin(), scored_frontiers.end(),
        [this](const ScoredFrontier& f) { return f.gain < min_publish_gain_; }),
      scored_frontiers.end());

    if (scored_frontiers.empty()) {
      RCLCPP_INFO(get_logger(), "All frontiers below min gain threshold (%.2f) - area likely explored", min_publish_gain_);
      return {};
    }

    // 6. Sort by gain (highest first)
    std::sort(scored_frontiers.begin(), scored_frontiers.end(),
              [](const ScoredFrontier& a, const ScoredFrontier& b) {
                return a.gain > b.gain;
              });

    // 7. Return top N frontiers
    std::vector<geometry_msgs::msg::Point> result;
    size_t count = std::min(num_frontiers, scored_frontiers.size());
    
    for (size_t i = 0; i < count; ++i) {
      geometry_msgs::msg::Point p;
      p.x = scored_frontiers[i].point.x();
      p.y = scored_frontiers[i].point.y();
      p.z = scored_frontiers[i].point.z();
      result.push_back(p);
      
      RCLCPP_INFO(get_logger(), "Frontier %zu: [%.1f, %.1f, %.1f] gain: %.2f",
                   i, p.x, p.y, p.z, scored_frontiers[i].gain);
    }

    return result;
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
        bool has_nearby_obstacle = false;

        for (auto& dir : directions) {
          octomap::point3d neighbor_point = 
            leaf_coordinate + octomap::point3d(dir[0], dir[1], dir[2]);
          octomap::OcTreeNode* neighbor_node = octree->search(neighbor_point);
          
          // If neighbor doesn't exist in tree, it's unknown → this is a frontier
          if (!neighbor_node) {
            is_frontier = true;
          } 
          // Check if neighbor is occupied (too close to wall)
          else if (neighbor_node->getOccupancy() > occupied_threshold_) {
            has_nearby_obstacle = true;
          }
        }
        
        // Only add frontier if it's valid and not directly adjacent to obstacles
        if (is_frontier && !has_nearby_obstacle) {
          frontiers.push_back(leaf_coordinate);
        }
      }
    }
    return frontiers;
  }

  float calculate_volumetric_gain(octomap::OcTree* octree, const octomap::point3d& point, float radius)
  {
    int unknown_voxels = 0;
    int total_voxels = 0;
    float voxel_size = octree->getResolution();
    float step_size = (voxel_size > 0.4f) ? voxel_size : 0.4f; // Optimization: Larger steps
    
    // Calculate volumetric exploration gain
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
    
    // Add penalty for being too close to obstacles (encourage center of free space)
    float min_obstacle_dist = 20.0f; // Start with large distance
    octomap::point3d end_point;
    
    // Check distance to obstacles in 26 directions (more robust than just cardinal)
    std::vector<octomap::point3d> directions;
    for(int x=-1; x<=1; x++) {
      for(int y=-1; y<=1; y++) {
        for(int z=-1; z<=1; z++) {
          if(x==0 && y==0 && z==0) continue;
          directions.push_back(octomap::point3d(x, y, z));
        }
      }
    }
    
    for (const auto& dir : directions) {
        // Cast ray to find nearest obstacle in each direction
        // Increased range to 20.0 to better find centers in large spaces
        if (octree->castRay(point, dir, end_point, true, 20.0)) {
            float dist = (end_point - point).norm();
            if (dist < min_obstacle_dist) {
                min_obstacle_dist = dist;
            }
        }
    }
    
    // Reward points that are further from walls (centered in passages)
    // Penalize frontiers close to walls (user requirement: < 4m)
    if (min_obstacle_dist < 4.0f) {
      return volume_gain * 0.001f; // Heavy penalty to effectively ignore them
    }

    // Use distance heavily in score to push towards center
    // We remove the cap at 1.0 so points further away get higher scores
    float centering_bonus = min_obstacle_dist;
    
    return volume_gain * centering_bonus;
  }

  /**
   * Calculate direction gain based on drone movement history
   * 
   * This calculates gain based on the drone's overall movement direction 
   * over the last N samples (moving average). This biases frontier selection 
   * toward where the drone has been heading, making exploration more coherent.
   * 
   * Returns [0, 1] where 1.0 = perfectly aligned with movement direction, 0.0 = opposite
   */
  float calculate_direction_gain(const octomap::point3d& frontier_point)
  {
    // If we don't have enough movement history, return neutral gain
    if (movement_history_.size() < 2) {
      return 0.5f;
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

    tf2::Vector3 drone_pos(last_drone_position_.x, last_drone_position_.y, last_drone_position_.z);
    tf2::Vector3 frontier_vec(frontier_point.x() - drone_pos.x(),
                              frontier_point.y() - drone_pos.y(),
                              frontier_point.z() - drone_pos.z());
    frontier_vec.normalize();

    // Dot product gives alignment: 1.0 = same direction, -1.0 = opposite
    float alignment = avg_movement.dot(frontier_vec);
    
    // Map from [-1, 1] to [0, 1]
    return (alignment + 1.0f) / 2.0f;
  }

  // ---------------------------------------------------------------------------
  // Publishing Helpers
  // ---------------------------------------------------------------------------

  void publish_markers(const std::vector<geometry_msgs::msg::Point>& frontiers)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    
    for (size_t i = 0; i < frontiers.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = this->get_clock()->now();
      marker.ns = "frontier_goals";
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = frontiers[i];
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0;
      marker.scale.y = 2.0;
      marker.scale.z = 2.0;
      marker.color.a = 0.8;
      
      // Color gradient: best (green) to worst (yellow)
      float ratio = static_cast<float>(i) / std::max(1.0, static_cast<double>(frontiers.size() - 1));
      marker.color.r = ratio;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      
      marker_array.markers.push_back(marker);
    }
    
    marker_publisher_->publish(marker_array);
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