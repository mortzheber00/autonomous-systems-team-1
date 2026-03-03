/**
 * @file global_planner_node.cpp
 * @brief State-Machine Global Planner for Autonomous Cave Exploration
 * @author Phil Chudalla (original), refactored for junction-based backtracking
 *
 * This node orchestrates the full exploration lifecycle as a state machine:
 *
 *   EXPLORING  ->  BACKTRACKING  ->  EXPLORING  ->  ...  ->  RETURNING_HOME
 *
 * Key responsibilities:
 *   1. Detect cave intersections (junctions) from diverging frontier clusters
 *   2. Record which junction arms have been explored
 *   3. When the local planner runs dry, select the best unexplored junction arm
 *      and command the RRT to fly there via /global_goal
 *   4. Once the cave is sufficiently explored, command return to home base
 *
 * ROS Interface:
 *   Subscribers:
 *     /frontier_goals      (PoseArray)   - frontiers from local planner
 *     /current_state_est   (Odometry)    - drone pose
 *     /octomap_full        (Octomap)     - occupancy map
 *     /rrt_status          (String)      - "idle" / "active" from RRT node
 *
 *   Publishers:
 *     /global_goal          (PoseArray)  - high-priority goal for RRT
 *     /global_frontier_markers (MarkerArray) - RViz visualization
 *     /exploration_state    (String)     - current state for monitoring
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <memory>
#include <chrono>
#include <string>
#include <queue>
#include <limits>
#include <std_msgs/msg/bool.hpp>

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Represents one unexplored arm of a junction.
 * The direction vector points from the junction center toward the frontier cluster.
 */
struct JunctionArm {
  Eigen::Vector3d direction;          ///< Unit direction of this arm
  geometry_msgs::msg::Point target;   ///< Representative frontier position
  float exploration_gain = 1.0f;      ///< Current gain (re-evaluated from OctoMap)
  bool explored = false;              ///< Whether the drone has taken this arm
};

/**
 * A cave intersection where multiple exploration paths diverge.
 * Detected when the local planner publishes frontiers in >=2 distinct angular clusters.
 */
struct Junction {
  int id;
  geometry_msgs::msg::Point position;             ///< Where the drone was when the junction was detected
  std::vector<JunctionArm> arms;                   ///< Diverging exploration directions
  rclcpp::Time last_updated;                       ///< Timestamp of last gain re-evaluation
  bool fully_explored = false;                     ///< True when all arms are explored or low-gain
  size_t creation_breadcrumb_idx = 0;              ///< Breadcrumb index when junction was created;
                                                   ///  only breadcrumbs AFTER this index are used
                                                   ///  for arm-explored detection

  Junction() : id(0), last_updated(0, 0, RCL_ROS_TIME) {}

  /// Count of arms that still have meaningful exploration potential
  int unexplored_arm_count(float gain_threshold = 0.1f) const {
    int count = 0;
    for (const auto& arm : arms) {
      if (!arm.explored && arm.exploration_gain > gain_threshold) {
        count++;
      }
    }
    return count;
  }

  /// Best unexplored arm (highest gain), or nullptr
  const JunctionArm* best_unexplored_arm() const {
    const JunctionArm* best = nullptr;
    float best_gain = -1.0f;
    for (const auto& arm : arms) {
      if (!arm.explored && arm.exploration_gain > best_gain) {
        best_gain = arm.exploration_gain;
        best = &arm;
      }
    }
    return best;
  }
};

// ============================================================================
// Exploration States
// ============================================================================
enum class ExplorationState {
  EXPLORING,        ///< Local planner is driving exploration forward
  BACKTRACKING,     ///< Flying back to a junction via RRT
  RETURNING_HOME,   ///< Mission complete, flying to home base
  DONE              ///< Arrived home, exploration finished
};

static std::string state_to_string(ExplorationState s) {
  switch (s) {
    case ExplorationState::EXPLORING:      return "EXPLORING";
    case ExplorationState::BACKTRACKING:   return "BACKTRACKING";
    case ExplorationState::RETURNING_HOME: return "RETURNING_HOME";
    case ExplorationState::DONE:           return "DONE";
  }
  return "UNKNOWN";
}

// ============================================================================
// GlobalPlannerNode
// ============================================================================
class GlobalPlannerNode : public rclcpp::Node
{
public:
  GlobalPlannerNode() : Node("global_planner_node")
  {
    // ---- Parameters ----
    declare_parameter("home_x", -360.0);
    declare_parameter("home_y",    0.0);
    declare_parameter("home_z",   16.0);
    declare_parameter("junction_detect_angle_deg", 60.0);
    declare_parameter("junction_proximity_m",      20.0);
    declare_parameter("no_frontier_timeout_sec",   12.0);
    declare_parameter("min_gain_threshold",         0.1);
    declare_parameter("goal_reached_tolerance_m",   8.0);
    declare_parameter("exploration_eval_radius_m", 50.0);
    declare_parameter("stagnation_timeout_sec",    20.0);
    declare_parameter("junction_confirm_count",       1);
    declare_parameter("min_frontier_gain_for_timer",  0.3);    
    declare_parameter("candidate_merge_radius_m",    30.0);
    declare_parameter("breadcrumb_spacing_m",        30.0);
    declare_parameter("wall_clearance_m",             3.0);
    declare_parameter("planning_loop_hz",              2.0);
    declare_parameter("idle_timeout_sec",             120.0);
    declare_parameter("idle_movement_radius_m",        2.0);
    declare_parameter("junction_max_vertical_z",       0.5);  // |z| above this = vertical arm, ignored
    home_position_.x = get_parameter("home_x").as_double();
    home_position_.y = get_parameter("home_y").as_double();
    home_position_.z = get_parameter("home_z").as_double();
    junction_angle_threshold_    = get_parameter("junction_detect_angle_deg").as_double() * M_PI / 180.0;
    junction_proximity_          = get_parameter("junction_proximity_m").as_double();
    no_frontier_timeout_         = get_parameter("no_frontier_timeout_sec").as_double();
    min_gain_threshold_          = get_parameter("min_gain_threshold").as_double();
    goal_reached_tolerance_      = get_parameter("goal_reached_tolerance_m").as_double();
    exploration_eval_radius_     = get_parameter("exploration_eval_radius_m").as_double();
    stagnation_timeout_          = get_parameter("stagnation_timeout_sec").as_double();
    junction_confirm_threshold_  = get_parameter("junction_confirm_count").as_int();
    min_frontier_gain_for_timer_ = get_parameter("min_frontier_gain_for_timer").as_double();
    candidate_merge_radius_      = get_parameter("candidate_merge_radius_m").as_double();
    breadcrumb_spacing_          = get_parameter("breadcrumb_spacing_m").as_double();
    wall_clearance_              = get_parameter("wall_clearance_m").as_double();
    planning_loop_hz_            = std::max(0.1, get_parameter("planning_loop_hz").as_double());
    idle_timeout_sec_            = get_parameter("idle_timeout_sec").as_double();
    idle_movement_radius_        = get_parameter("idle_movement_radius_m").as_double();
    junction_max_vertical_z_     = get_parameter("junction_max_vertical_z").as_double();

    // ---- Publishers ----
    global_goal_pub_    = create_publisher<geometry_msgs::msg::PoseArray>("/global_goal", 10);
    state_pub_          = create_publisher<std_msgs::msg::String>("/exploration_state", 10);
    marker_pub_         = create_publisher<visualization_msgs::msg::MarkerArray>("/global_frontier_markers", 10);

    // ---- Subscribers ----
    frontier_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/frontier_goals", 10,
      std::bind(&GlobalPlannerNode::frontier_callback, this, std::placeholders::_1));

    octomap_sub_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_full", rclcpp::QoS(10).best_effort(),
      std::bind(&GlobalPlannerNode::octomap_callback, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/current_state_est", 10,
      std::bind(&GlobalPlannerNode::odom_callback, this, std::placeholders::_1));

    rrt_status_sub_ = create_subscription<std_msgs::msg::String>(
      "/rrt_status", 10,
      std::bind(&GlobalPlannerNode::rrt_status_callback, this, std::placeholders::_1));

    // Added this for: Enable/Disable Local Planner for state machine control
    enable_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/state_machine/enable_global_planner", 1,
      std::bind(&GlobalPlannerNode::enable_callback, this, std::placeholders::_1)
    );

    // ---- Timers ----
    const auto planning_period = std::chrono::duration<double>(1.0 / planning_loop_hz_);
    planning_timer_ = create_wall_timer(
      planning_period,
      std::bind(&GlobalPlannerNode::planning_loop, this));

    viz_timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&GlobalPlannerNode::publish_markers, this));

    last_frontier_time_ = now();

    RCLCPP_INFO(get_logger(),
      "Global Planner initialized. Home=(%.0f, %.0f, %.0f), "
      "junction_angle=%.0f deg, timeout=%.0fs, planning_loop=%.1f Hz",
      home_position_.x, home_position_.y, home_position_.z,
      junction_angle_threshold_ * 180.0 / M_PI, no_frontier_timeout_, planning_loop_hz_);
  }

private:
  // ---- State ----
  ExplorationState state_ = ExplorationState::EXPLORING;
  geometry_msgs::msg::Point current_position_;
  geometry_msgs::msg::Point home_position_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_subscription_;
  bool odom_received_ = false;

  // State Machine for Goal Persistence
  bool global_planner_enabled_ = false;

  // Junction management
  std::vector<Junction> junctions_;
  int junction_counter_ = 0;

  // Frontier tracking
  rclcpp::Time last_frontier_time_;
  std::vector<geometry_msgs::msg::Point> latest_frontiers_;

  // RRT feedback
  std::string rrt_status_ = "idle";

  // Backtrack target
  geometry_msgs::msg::Point backtrack_target_;
  int backtrack_junction_id_ = -1;

  // Breadcrumb trail: record drone positions for backtrack routing
  std::vector<geometry_msgs::msg::Point> breadcrumbs_;
  double breadcrumb_spacing_ = 30.0;  ///< Min distance between breadcrumbs

  // Multi-hop backtrack waypoint queue (short hops the RRT can actually reach)
  std::vector<geometry_msgs::msg::Point> backtrack_waypoints_;
  size_t backtrack_wp_idx_ = 0;

  // Exploration progress tracking
  double last_explored_volume_ = 0.0;     ///< Number of known (free+occupied) leaves last check
  rclcpp::Time last_progress_time_;        ///< Last time we saw meaningful exploration progress
  bool progress_time_initialized_ = false;
  double stagnation_timeout_ = 20.0;       ///< Seconds of no new exploration before triggering backtrack
  double min_volume_delta_ = 5.0;          ///< Minimum new leaves to count as "progress"

  // Junction candidate confirmation (temporal filtering)
  struct JunctionCandidate {
    geometry_msgs::msg::Point position;       ///< First-seen position (merge anchor)
    geometry_msgs::msg::Point position_sum;   ///< Running sum for averaging across detections
    int position_count = 0;                   ///< Number of detections included in sum
    int detection_count = 0;
    rclcpp::Time first_seen;
    rclcpp::Time last_seen;
    std::vector<std::vector<geometry_msgs::msg::Point>> arm_clusters; // saved for creation
    std::vector<Eigen::Vector3d> arm_directions;
  };
  std::vector<JunctionCandidate> junction_candidates_;
  int junction_confirm_threshold_ = 3;     ///< Detections needed to confirm a junction
  double junction_candidate_timeout_ = 10.0; ///< Seconds before a candidate expires

  // OctoMap
  std::shared_ptr<octomap::OcTree> octree_;

  // Configuration
  double junction_angle_threshold_;
  double junction_proximity_;
  double no_frontier_timeout_;
  double min_gain_threshold_;
  double goal_reached_tolerance_;
  double exploration_eval_radius_;
  double min_frontier_gain_for_timer_;  ///< Only frontiers with gain above this reset the timer
  double candidate_merge_radius_;       ///< Radius for merging junction candidate detections
  double wall_clearance_;                ///< Min distance a backtrack target must be from walls
  double planning_loop_hz_ = 2.0;        ///< Frequency of planning loop updates
  double idle_timeout_sec_ = 120.0;       ///< Seconds of no motion before forcing return home
  double idle_movement_radius_ = 2.0;    ///< Distance that counts as motion progress
  double junction_max_vertical_z_; ///< Cluster arms with |z| > this are considered vertical (top/bottom of cave) and skipped
  geometry_msgs::msg::Point last_movement_position_{};
  rclcpp::Time last_movement_time_;
  bool movement_tracking_initialized_ = false;

  // ---- ROS handles ----
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr global_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr frontier_sub_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr rrt_status_sub_;

  rclcpp::TimerBase::SharedPtr planning_timer_;
  rclcpp::TimerBase::SharedPtr viz_timer_;

  // ==========================================================================
  // Callbacks
  // ==========================================================================

  void enable_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    global_planner_enabled_ = msg->data;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_position_ = msg->pose.pose.position;
    odom_received_ = true;

    const auto now_t = now();
    if (!movement_tracking_initialized_) {
      last_movement_position_ = current_position_;
      last_movement_time_ = now_t;
      movement_tracking_initialized_ = true;
    } else if (point_distance(current_position_, last_movement_position_) >= idle_movement_radius_) {
      last_movement_position_ = current_position_;
      last_movement_time_ = now_t;
    }

    // Record breadcrumbs for backtrack routing and arm-explored detection.
    // Recorded in ALL states so that backtracking flights also leave a trail
    // (needed to mark junction arms the drone passes through during backtrack).
    if (breadcrumbs_.empty() ||
        point_distance(current_position_, breadcrumbs_.back()) >= breadcrumb_spacing_) {
      breadcrumbs_.push_back(current_position_);
    }
  }

  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
    if (tree) {
      auto* ot = dynamic_cast<octomap::OcTree*>(tree);
      if (ot) {
        octree_ = std::shared_ptr<octomap::OcTree>(ot);
      } else {
        delete tree;
      }
    }
  }

  void rrt_status_callback(const std_msgs::msg::String::SharedPtr msg) {
    rrt_status_ = msg->data;
  }

  /**
   * Called every time the local planner publishes frontiers.
   *
   * CRITICAL FIX: Only resets the "no frontier" timer for high-quality frontiers.
   * Low-gain frontiers (dead-end noise, loop artifacts) are ignored for the
   * purpose of stalling detection, which allows BACKTRACKING to trigger.
   *
   * Frontiers are also gain-validated before being considered for junction detection
   * to prevent spurious junctions from marginal frontier cells.
   */
  void frontier_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    if (msg->poses.empty()) return;

    // Store latest frontiers
    latest_frontiers_.clear();
    for (const auto& pose : msg->poses) {
      latest_frontiers_.push_back(pose.position);
    }

    // Evaluate gain of incoming frontiers against the OctoMap.
    // Only reset the timer if at least one frontier has meaningful gain.
    std::vector<geometry_msgs::msg::Point> high_quality_frontiers;
    if (octree_) {
      for (const auto& f : latest_frontiers_) {
        float gain = calculate_exploration_gain(f);
        if (gain >= min_frontier_gain_for_timer_) {
          high_quality_frontiers.push_back(f);
        }
      }
    } else {
      // No OctoMap yet — trust the local planner
      high_quality_frontiers = latest_frontiers_;
    }

    if (!high_quality_frontiers.empty()) {
      last_frontier_time_ = now();
      RCLCPP_DEBUG(get_logger(), "[GLOBAL] %zu/%zu frontiers above gain threshold — timer reset",
        high_quality_frontiers.size(), latest_frontiers_.size());
    } else {
      RCLCPP_DEBUG(get_logger(), "[GLOBAL] All %zu frontiers below gain threshold — timer NOT reset",
        latest_frontiers_.size());
    }

    // While exploring, try to detect junctions using only high-quality frontiers
    if (state_ == ExplorationState::EXPLORING && odom_received_ && high_quality_frontiers.size() >= 2) {
      try_detect_junction(high_quality_frontiers);
    }
  }

  // ==========================================================================
  // Main Planning Loop (1 Hz)
  // ==========================================================================

  void planning_loop() {
    if (!odom_received_) return;

    // enable by state-machine
    if (!global_planner_enabled_) {
        return;
    }

    // Publish current state for monitoring
    std_msgs::msg::String state_msg;
    state_msg.data = state_to_string(state_);
    state_pub_->publish(state_msg);

    switch (state_) {
      case ExplorationState::EXPLORING:
        handle_exploring();
        break;
      case ExplorationState::BACKTRACKING:
        handle_backtracking();
        break;
      case ExplorationState::RETURNING_HOME:
        handle_returning_home();
        break;
      case ExplorationState::DONE:
        break;
    }

    check_idle_timeout();

    // Periodically re-evaluate junction gains
    re_evaluate_junction_gains();
  }

  // ==========================================================================
  // State Handlers
  // ==========================================================================

  /**
   * EXPLORING: local planner is in charge.
   * Transition to BACKTRACKING when no frontiers for no_frontier_timeout_ seconds.
   */
  void handle_exploring() {
    double elapsed = (now() - last_frontier_time_).seconds();

    // Mark junction arms as explored when we pass near them
    mark_explored_arms();

    // Track exploration progress via OctoMap volume
    check_exploration_progress();

    // Also check stagnation: even if low-quality frontiers keep arriving,
    // if the map hasn't grown in stagnation_timeout_ seconds, trigger backtrack
    double stagnation_elapsed = 0.0;
    if (progress_time_initialized_) {
      stagnation_elapsed = (now() - last_progress_time_).seconds();
    }
    bool is_stagnating = progress_time_initialized_ && (stagnation_elapsed > stagnation_timeout_);

    if (is_stagnating) {
      RCLCPP_WARN(get_logger(),
        "[GLOBAL] Exploration stagnating for %.1fs — no new map volume discovered.", stagnation_elapsed);
    }

    if (elapsed > no_frontier_timeout_ || is_stagnating) {
      RCLCPP_WARN(get_logger(),
        "[GLOBAL] No frontiers for %.1fs - local planner exhausted.", elapsed);

      // Find best unexplored junction
      const Junction* best_jct = nullptr;
      const JunctionArm* best_arm = nullptr;
      float best_gain = -1.0f;

      for (const auto& jct : junctions_) {
        if (jct.fully_explored) continue;
        const JunctionArm* arm = jct.best_unexplored_arm();
        if (arm && arm->exploration_gain > best_gain) {
          best_gain = arm->exploration_gain;
          best_arm = arm;
          best_jct = &jct;
        }
      }

      if (best_jct && best_arm && best_gain > min_gain_threshold_) {
        // Build multi-hop waypoint route from breadcrumbs to the junction
        // Ensure the arm target is not dangerously close to a wall.
        // If it is, pull it toward the junction centre (a known-safe position).
        backtrack_target_ = adjust_for_wall_clearance(
            best_arm->target, best_jct->position, wall_clearance_);
        backtrack_junction_id_ = best_jct->id;
        backtrack_waypoints_ = build_backtrack_route(best_jct->position);
        // Add the (possibly adjusted) arm target as the final waypoint
        backtrack_waypoints_.push_back(backtrack_target_);
        backtrack_wp_idx_ = 0;
        state_ = ExplorationState::BACKTRACKING;

        RCLCPP_INFO(get_logger(),
          "[GLOBAL] -> BACKTRACKING to junction %d via %zu waypoints, final target (%.1f, %.1f, %.1f) gain=%.2f",
          best_jct->id, backtrack_waypoints_.size(),
          backtrack_target_.x, backtrack_target_.y, backtrack_target_.z,
          best_gain);

        if (!backtrack_waypoints_.empty()) {
          publish_global_goal(backtrack_waypoints_[0]);
        }
      } else {
        // No junctions left -> go home via breadcrumb route
        initiate_return_home("no junctions remaining");
      }
    }
  }

  /**
   * BACKTRACKING: Follow multi-hop waypoints back to a junction.
   * Each hop is short enough for the RRT to handle.
   */
  void handle_backtracking() {
    // Also mark arms during backtracking — the drone may fly through or
    // past other junctions on its way to the target junction.
    mark_explored_arms();

    if (backtrack_waypoints_.empty() || backtrack_wp_idx_ >= backtrack_waypoints_.size()) {
      // Shouldn't happen, but recover gracefully
      RCLCPP_WARN(get_logger(), "[GLOBAL] Backtrack waypoints exhausted -> EXPLORING");
      state_ = ExplorationState::EXPLORING;
      last_frontier_time_ = now();
      return;
    }

    auto& current_wp = backtrack_waypoints_[backtrack_wp_idx_];
    double dist = point_distance(current_position_, current_wp);

    if (dist < goal_reached_tolerance_) {
      RCLCPP_INFO(get_logger(),
        "[GLOBAL] Reached backtrack waypoint %zu/%zu (dist=%.1fm)",
        backtrack_wp_idx_ + 1, backtrack_waypoints_.size(), dist);

      backtrack_wp_idx_++;

      if (backtrack_wp_idx_ >= backtrack_waypoints_.size()) {
        // Arrived at final target — mark arm explored and resume
        RCLCPP_INFO(get_logger(),
          "[GLOBAL] Reached junction %d -> EXPLORING", backtrack_junction_id_);

        for (auto& jct : junctions_) {
          if (jct.id == backtrack_junction_id_) {
            for (auto& arm : jct.arms) {
              double arm_dist = std::sqrt(
                std::pow(arm.target.x - backtrack_target_.x, 2) +
                std::pow(arm.target.y - backtrack_target_.y, 2) +
                std::pow(arm.target.z - backtrack_target_.z, 2));
              if (arm_dist < 5.0) {
                arm.explored = true;
              }
            }
            jct.fully_explored = (jct.unexplored_arm_count() == 0);
          }
        }

        state_ = ExplorationState::EXPLORING;
        last_frontier_time_ = now();
        // Reset stagnation timer so we get a fresh window
        progress_time_initialized_ = false;
      } else {
        // Publish next waypoint
        publish_global_goal(backtrack_waypoints_[backtrack_wp_idx_]);
      }
    } else {
      // Re-publish current waypoint so RRT keeps pursuing it
      publish_global_goal(current_wp);
    }
  }

  /**
   * RETURNING_HOME: Follow multi-hop waypoints back to home base.
   */
  void handle_returning_home() {
    if (backtrack_waypoints_.empty() || backtrack_wp_idx_ >= backtrack_waypoints_.size()) {
      // Direct distance check as fallback
      double dist = point_distance(current_position_, home_position_);
      if (dist < goal_reached_tolerance_) {
        RCLCPP_WARN(get_logger(),
          "[GLOBAL] Arrived home (dist=%.1fm). Exploration complete!", dist);
        state_ = ExplorationState::DONE;
      } else {
        publish_global_goal(home_position_);
      }
      return;
    }

    auto& current_wp = backtrack_waypoints_[backtrack_wp_idx_];
    double dist = point_distance(current_position_, current_wp);

    if (dist < goal_reached_tolerance_) {
      RCLCPP_INFO(get_logger(),
        "[GLOBAL] Reached home waypoint %zu/%zu (dist=%.1fm)",
        backtrack_wp_idx_ + 1, backtrack_waypoints_.size(), dist);

      backtrack_wp_idx_++;

      if (backtrack_wp_idx_ >= backtrack_waypoints_.size()) {
        RCLCPP_WARN(get_logger(),
          "[GLOBAL] Arrived home. Exploration complete!");
        state_ = ExplorationState::DONE;
      } else {
        publish_global_goal(backtrack_waypoints_[backtrack_wp_idx_]);
      }
    } else {
      publish_global_goal(current_wp);
    }
  }

  void initiate_return_home(const std::string& reason) {
    backtrack_waypoints_ = build_backtrack_route(home_position_);
    backtrack_waypoints_.push_back(home_position_);
    backtrack_wp_idx_ = 0;
    state_ = ExplorationState::RETURNING_HOME;
    RCLCPP_WARN(get_logger(),
      "[GLOBAL] -> RETURNING_HOME (%s) via %zu waypoints",
      reason.c_str(), backtrack_waypoints_.size());

    if (!backtrack_waypoints_.empty()) {
      publish_global_goal(backtrack_waypoints_[0]);
    } else {
      publish_global_goal(home_position_);
    }
  }

  void check_idle_timeout() {
    if (!global_planner_enabled_) return;
    if (state_ == ExplorationState::RETURNING_HOME || state_ == ExplorationState::DONE) return;
    if (!movement_tracking_initialized_) return;

    double idle_elapsed = (now() - last_movement_time_).seconds();
    if (idle_elapsed > idle_timeout_sec_) {
      RCLCPP_WARN(get_logger(),
        "[GLOBAL] No motion > %.1fs within %.1fm — forcing return home",
        idle_elapsed, idle_movement_radius_);
      initiate_return_home("motion timeout");
      progress_time_initialized_ = false;
    }
  }

  // ==========================================================================
  // Junction Detection
  // ==========================================================================

  /**
   * Detect if current frontiers represent a cave intersection.
   *
   * Algorithm:
   *   1. Compute direction vectors from drone to each frontier
   *   2. Cluster directions by angular proximity
   *   3. If >=2 clusters with angular separation > threshold -> junction
   *   4. Check we have not already recorded a junction nearby
   */
  /**
   * Detect if current frontiers represent a cave intersection.
   *
   * Improvements over naive approach:
   *   - Only considers frontiers with validated OctoMap gain (pre-filtered by caller)
   *   - Requires minimum distance from drone (filters out close scatter)
   *   - Uses TEMPORAL CONFIRMATION: a junction must be detected N times
   *     at a consistent location before it is registered
   *   - Stale candidates are cleaned up
   */
  void try_detect_junction(const std::vector<geometry_msgs::msg::Point>& frontiers) {
    if (frontiers.size() < 2) return;

    // 1. Compute direction vectors, requiring minimum distance from drone
    const double min_frontier_dist = 8.0;  // Ignore frontiers closer than this
    struct DirFrontier {
      Eigen::Vector3d dir;
      geometry_msgs::msg::Point point;
    };
    std::vector<DirFrontier> dir_frontiers;

    for (const auto& f : frontiers) {
      Eigen::Vector3d d(
        f.x - current_position_.x,
        f.y - current_position_.y,
        f.z - current_position_.z);
      double len = d.norm();
      if (len > min_frontier_dist) {
        d.normalize();
        dir_frontiers.push_back({d, f});
      }
    }

    if (dir_frontiers.size() < 2) return;

    // 2. Greedy angular clustering
    std::vector<std::vector<DirFrontier>> clusters;
    std::vector<bool> assigned(dir_frontiers.size(), false);

    for (size_t i = 0; i < dir_frontiers.size(); ++i) {
      if (assigned[i]) continue;
      std::vector<DirFrontier> cluster = {dir_frontiers[i]};
      assigned[i] = true;

      for (size_t j = i + 1; j < dir_frontiers.size(); ++j) {
        if (assigned[j]) continue;
        double angle = std::acos(
          std::clamp(dir_frontiers[i].dir.dot(dir_frontiers[j].dir), -1.0, 1.0));
        if (angle < junction_angle_threshold_) {
          cluster.push_back(dir_frontiers[j]);
          assigned[j] = true;
        }
      }
      clusters.push_back(cluster);
    }

    if (clusters.size() < 2) return;

    // 3. Verify angular separation between cluster centroids
    std::vector<Eigen::Vector3d> centroids;
    for (const auto& cluster : clusters) {
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      for (const auto& df : cluster) centroid += df.dir;
      centroid.normalize();
      centroids.push_back(centroid);
    }

    // 3b. Filter out vertical arms (pointing toward ceiling/floor of the cave).
    //     Junctions are only left/right (horizontal) — arms whose centroid
    //     direction has |z| > junction_max_vertical_z_ are discarded.
    {
      std::vector<Eigen::Vector3d> h_centroids;
      std::vector<std::vector<DirFrontier>> h_clusters;
      for (size_t i = 0; i < centroids.size(); ++i) {
        if (std::abs(centroids[i].z()) <= junction_max_vertical_z_) {
          h_centroids.push_back(centroids[i]);
          h_clusters.push_back(clusters[i]);
        } else {
          RCLCPP_DEBUG(get_logger(),
            "[JUNCTION] Dropping vertical arm dir=(%.2f,%.2f,%.2f) — top/bottom of cave",
            centroids[i].x(), centroids[i].y(), centroids[i].z());
        }
      }
      if (h_centroids.size() < 2) return;  // Not enough horizontal arms left
      centroids = std::move(h_centroids);
      clusters  = std::move(h_clusters);
    }

    bool has_wide_spread = false;
    for (size_t i = 0; i < centroids.size() && !has_wide_spread; ++i) {
      for (size_t j = i + 1; j < centroids.size(); ++j) {
        double angle = std::acos(std::clamp(centroids[i].dot(centroids[j]), -1.0, 1.0));
        if (angle > junction_angle_threshold_) {
          has_wide_spread = true;
          break;
        }
      }
    }
    if (!has_wide_spread) return;

    // 4. Check proximity to already-confirmed junctions
    for (const auto& jct : junctions_) {
      if (point_distance(jct.position, current_position_) < junction_proximity_) {
        return;  // Already registered
      }
    }

    // 5. TEMPORAL CONFIRMATION: find or create a candidate
    //    Clean up stale candidates first
    auto now_t = now();
    junction_candidates_.erase(
      std::remove_if(junction_candidates_.begin(), junction_candidates_.end(),
        [&](const JunctionCandidate& c) {
          return (now_t - c.last_seen).seconds() > junction_candidate_timeout_;
        }),
      junction_candidates_.end());

    // Look for existing candidate near current position
    // Use candidate_merge_radius_ (larger) so detections while the drone
    // is moving through a junction area still confirm the same candidate.
    JunctionCandidate* match = nullptr;
    double best_match_dist = candidate_merge_radius_;
    for (auto& cand : junction_candidates_) {
      double d = point_distance(cand.position, current_position_);
      if (d < best_match_dist) {
        best_match_dist = d;
        match = &cand;
      }
    }

    if (match) {
      match->detection_count++;
      match->last_seen = now_t;
      // Accumulate position for averaging
      match->position_sum.x += current_position_.x;
      match->position_sum.y += current_position_.y;
      match->position_sum.z += current_position_.z;
      match->position_count++;
      // Update arm directions with latest data
      match->arm_directions.clear();
      for (const auto& c : centroids) match->arm_directions.push_back(c);
      match->arm_clusters.clear();
      for (const auto& cluster : clusters) {
        std::vector<geometry_msgs::msg::Point> pts;
        for (const auto& df : cluster) pts.push_back(df.point);
        match->arm_clusters.push_back(pts);
      }

      RCLCPP_DEBUG(get_logger(),
        "[JUNCTION CANDIDATE] at (%.1f, %.1f, %.1f) seen %d/%d times",
        match->position.x, match->position.y, match->position.z,
        match->detection_count, junction_confirm_threshold_);

      if (match->detection_count < junction_confirm_threshold_) {
        return;  // Not yet confirmed
      }
      // Fall through to create confirmed junction from this candidate
    } else {
      // Create new candidate
      JunctionCandidate cand;
      cand.position = current_position_;
      cand.position_sum = current_position_;
      cand.position_count = 1;
      cand.detection_count = 1;
      cand.first_seen = now_t;
      cand.last_seen = now_t;
      for (const auto& c : centroids) cand.arm_directions.push_back(c);
      for (const auto& cluster : clusters) {
        std::vector<geometry_msgs::msg::Point> pts;
        for (const auto& df : cluster) pts.push_back(df.point);
        cand.arm_clusters.push_back(pts);
      }
      junction_candidates_.push_back(cand);

      RCLCPP_INFO(get_logger(),
        "[JUNCTION CANDIDATE] New candidate at (%.1f, %.1f, %.1f) with %zu arms — needs %d confirmations",
        current_position_.x, current_position_.y, current_position_.z,
        clusters.size(), junction_confirm_threshold_);

      if (cand.detection_count < junction_confirm_threshold_) {
        return;  // Not yet confirmed
      }
      // Threshold is 0 or 1 — confirm immediately
      match = &junction_candidates_.back();
    }

    // 6. Confirmed — create the junction from the candidate
    Junction jct;
    jct.id = junction_counter_++;
    // Use the averaged drone position across all confirmations as the junction center.
    // This is more robust than anchoring to the first detection, which may have
    // occurred while the drone was still approaching the junction.
    {
      double n = static_cast<double>(match->position_count);
      jct.position.x = match->position_sum.x / n;
      jct.position.y = match->position_sum.y / n;
      jct.position.z = match->position_sum.z / n;
    }
    jct.last_updated = now_t;
    jct.creation_breadcrumb_idx = breadcrumbs_.size();  // only future breadcrumbs count

    for (size_t ci = 0; ci < match->arm_clusters.size(); ++ci) {
      JunctionArm arm;
      arm.direction = (ci < match->arm_directions.size()) ? match->arm_directions[ci] : Eigen::Vector3d::UnitX();

      // Use the centroid of the cluster's frontier points as the raw target.
      // Frontier points sit at the free/occupied boundary (wall edge), so we
      // immediately pull the target away from walls toward the junction center.
      geometry_msgs::msg::Point raw_target;
      const auto& cluster_pts = match->arm_clusters[ci];
      for (const auto& p : cluster_pts) {
        raw_target.x += p.x;
        raw_target.y += p.y;
        raw_target.z += p.z;
      }
      double n = static_cast<double>(cluster_pts.size());
      raw_target.x /= n;
      raw_target.y /= n;
      raw_target.z /= n;

      arm.target = adjust_for_wall_clearance(raw_target, jct.position, wall_clearance_);
      arm.exploration_gain = 1.0f;
      arm.explored = false;
      jct.arms.push_back(arm);
    }

    junctions_.push_back(jct);

    // Remove the candidate
    junction_candidates_.erase(
      std::remove_if(junction_candidates_.begin(), junction_candidates_.end(),
        [&](const JunctionCandidate& c) {
          return point_distance(c.position, match->position) < junction_proximity_;
        }),
      junction_candidates_.end());

    RCLCPP_INFO(get_logger(),
      "[JUNCTION] CONFIRMED junction %d at (%.1f, %.1f, %.1f) with %zu arms",
      jct.id, jct.position.x, jct.position.y, jct.position.z, jct.arms.size());
    for (size_t i = 0; i < jct.arms.size(); ++i) {
      RCLCPP_INFO(get_logger(),
        "  Arm %zu: dir=(%.2f, %.2f, %.2f) target=(%.1f, %.1f, %.1f)",
        i, jct.arms[i].direction.x(), jct.arms[i].direction.y(), jct.arms[i].direction.z(),
        jct.arms[i].target.x, jct.arms[i].target.y, jct.arms[i].target.z);
    }
  }

  // ==========================================================================
  // Junction Maintenance
  // ==========================================================================

  /**
   * Build the shortest route from the current position to a target using
   * the breadcrumb trail as a graph.
   *
   * The naive approach (walk the chain) fails when the drone explored a loop:
   * it would trace the entire loop perimeter.  Instead we run Dijkstra over a
   * graph where:
   *   - Consecutive breadcrumbs are always connected (the chain edges).
   *   - Any two breadcrumbs within `shortcut_radius` of each other are also
   *     connected (shortcut edges that cut across loops).
   *
   * Result: the shortest Euclidean-distance path through known positions.
   */
  std::vector<geometry_msgs::msg::Point> build_backtrack_route(
      const geometry_msgs::msg::Point& target) {

    std::vector<geometry_msgs::msg::Point> route;
    if (breadcrumbs_.empty()) {
      route.push_back(target);
      return route;
    }

    const size_t N = breadcrumbs_.size();

    // Find breadcrumb closest to the drone's current position
    size_t start_idx = 0;
    {
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i < N; ++i) {
        double d = point_distance(breadcrumbs_[i], current_position_);
        if (d < best) { best = d; start_idx = i; }
      }
    }

    // Find breadcrumb closest to the target
    size_t goal_idx = 0;
    {
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i < N; ++i) {
        double d = point_distance(breadcrumbs_[i], target);
        if (d < best) { best = d; goal_idx = i; }
      }
    }

    if (start_idx == goal_idx) {
      route.push_back(breadcrumbs_[goal_idx]);
      RCLCPP_INFO(get_logger(),
        "[BACKTRACK ROUTE] Start and goal on same breadcrumb");
      return route;
    }

    // ---- Dijkstra over breadcrumb graph ----
    // Shortcut radius: connect non-consecutive breadcrumbs that are physically
    // close. 1.5× spacing catches loop-closure shortcuts reliably.
    const double shortcut_radius = breadcrumb_spacing_ * 1.5;

    std::vector<double> dist(N, std::numeric_limits<double>::max());
    std::vector<int> prev(N, -1);

    // Min-heap: (distance, index)
    using PQEntry = std::pair<double, size_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    dist[start_idx] = 0.0;
    pq.push({0.0, start_idx});

    while (!pq.empty()) {
      auto [d_u, u] = pq.top();
      pq.pop();

      if (d_u > dist[u]) continue;   // stale entry
      if (u == goal_idx) break;      // shortest path found

      // Helper: relax edge u -> v
      auto relax = [&](size_t v) {
        double edge_len = point_distance(breadcrumbs_[u], breadcrumbs_[v]);
        double new_dist = dist[u] + edge_len;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          prev[v] = static_cast<int>(u);
          pq.push({new_dist, v});
        }
      };

      // 1. Chain edges (consecutive breadcrumbs)
      if (u > 0)      relax(u - 1);
      if (u + 1 < N)  relax(u + 1);

      // 2. Shortcut edges to any nearby breadcrumb (handles loops)
      for (size_t v = 0; v < N; ++v) {
        if (v == u) continue;
        if (v + 1 == u || u + 1 == v) continue;  // already handled
        if (point_distance(breadcrumbs_[u], breadcrumbs_[v]) <= shortcut_radius) {
          relax(v);
        }
      }
    }

    // ---- Reconstruct path ----
    if (prev[goal_idx] == -1 && start_idx != goal_idx) {
      RCLCPP_WARN(get_logger(),
        "[BACKTRACK ROUTE] Dijkstra found no path — using direct route");
      route.push_back(target);
      return route;
    }

    std::vector<size_t> path_indices;
    for (int idx = static_cast<int>(goal_idx); idx != -1; idx = prev[idx]) {
      path_indices.push_back(static_cast<size_t>(idx));
    }
    std::reverse(path_indices.begin(), path_indices.end());

    for (size_t idx : path_indices) {
      route.push_back(breadcrumbs_[idx]);
    }

    // Downsample: skip intermediate waypoints that are very close
    if (route.size() > 2) {
      std::vector<geometry_msgs::msg::Point> sparse;
      sparse.push_back(route.front());
      for (size_t i = 1; i < route.size() - 1; ++i) {
        if (point_distance(route[i], sparse.back()) >= breadcrumb_spacing_ * 0.8) {
          sparse.push_back(route[i]);
        }
      }
      sparse.push_back(route.back());
      route = sparse;
    }

    double total_dist = dist[goal_idx];
    RCLCPP_INFO(get_logger(),
      "[BACKTRACK ROUTE] Dijkstra: %zu waypoints, %.0fm total (from %zu breadcrumbs)",
      route.size(), total_dist, N);

    return route;
  }

  /**
   * Check if the OctoMap is still growing (exploration is making progress).
   * If the map volume hasn't increased in stagnation_timeout_ seconds,
   * the drone is likely looping or stuck in a dead end.
   */
  void check_exploration_progress() {
    if (!octree_) return;

    // Count total known leaves in the OctoMap
    double current_volume = static_cast<double>(octree_->getNumLeafNodes());

    if (!progress_time_initialized_) {
      last_explored_volume_ = current_volume;
      last_progress_time_ = now();
      progress_time_initialized_ = true;
      return;
    }

    double delta = current_volume - last_explored_volume_;
    if (delta > min_volume_delta_) {
      // Map is growing — exploration is progressing
      last_explored_volume_ = current_volume;
      last_progress_time_ = now();
      RCLCPP_DEBUG(get_logger(), "[PROGRESS] Map grew by %.0f leaves (total: %.0f)", delta, current_volume);
    }
  }

  /**
   * Mark junction arms as explored when the drone flies close to them
   * OR has traveled along the arm's direction (detected via breadcrumbs).
   *
   * Three checks per arm:
   *   1. Current position near the arm's original target point (legacy check)
   *   2. Any breadcrumb lies along the arm's direction from the junction centre
   *      at a sufficient distance → the drone has flown down that corridor
   *   3. Current position lies along the arm's direction (catches the case
   *      where the drone returns to a junction from a loop)
   */
  void mark_explored_arms() {
    // Cone parameters: a point must be within the angular cone AND within
    // the distance band [min, max] from the junction centre to count.
    // Beyond max the cave has likely turned and direction is unreliable.
    const double arm_angle_threshold = M_PI / 7.0;   //
    const double min_travel_from_jct = 40.0;          // Too close = still at junction
    const double max_travel_from_jct = 80.0;          // Too far = direction unreliable

    for (auto& jct : junctions_) {
      if (jct.fully_explored) continue;

      Eigen::Vector3d jct_pos(jct.position.x, jct.position.y, jct.position.z);

      for (auto& arm : jct.arms) {
        if (arm.explored) continue;

        // --- Check 1: proximity to original target (quick, cheap) ---
        double dist_to_target = point_distance(current_position_, arm.target);
        if (dist_to_target < goal_reached_tolerance_) {
          arm.explored = true;
          RCLCPP_INFO(get_logger(),
            "[JUNCTION] Marked arm of junction %d as explored "
            "(near target, dist=%.1fm)",
            jct.id, dist_to_target);
          continue;
        }

        // --- Check 2: breadcrumbs in the arm's cone ---
        //     ONLY breadcrumbs recorded AFTER the junction was created.
        //     Must be within the distance band [min, max] from junction.
        for (size_t bi = jct.creation_breadcrumb_idx; bi < breadcrumbs_.size(); ++bi) {
          Eigen::Vector3d crumb_pos(breadcrumbs_[bi].x, breadcrumbs_[bi].y, breadcrumbs_[bi].z);
          Eigen::Vector3d to_crumb = crumb_pos - jct_pos;
          double dist_from_jct = to_crumb.norm();

          // Skip if outside the detection band
          if (dist_from_jct < min_travel_from_jct || dist_from_jct > max_travel_from_jct)
            continue;

          to_crumb.normalize();
          double angle = std::acos(
            std::clamp(to_crumb.dot(arm.direction), -1.0, 1.0));

          if (angle < arm_angle_threshold) {
            arm.explored = true;
            RCLCPP_INFO(get_logger(),
              "[JUNCTION] Marked arm of junction %d as explored "
              "(breadcrumb in cone, angle=%.1f°, dist=%.1fm from jct)",
              jct.id, angle * 180.0 / M_PI, dist_from_jct);
            break;
          }
        }
        if (arm.explored) continue;

        // --- Check 3: current position in the arm's cone ---
        //     Only after the drone has moved away since junction creation.
        if (breadcrumbs_.size() <= jct.creation_breadcrumb_idx) continue;

        Eigen::Vector3d cur_pos(current_position_.x,
                                current_position_.y,
                                current_position_.z);
        Eigen::Vector3d to_cur = cur_pos - jct_pos;
        double dist_cur = to_cur.norm();

        if (dist_cur >= min_travel_from_jct && dist_cur <= max_travel_from_jct) {
          to_cur.normalize();
          double angle = std::acos(
            std::clamp(to_cur.dot(arm.direction), -1.0, 1.0));

          if (angle < arm_angle_threshold) {
            arm.explored = true;
            RCLCPP_INFO(get_logger(),
              "[JUNCTION] Marked arm of junction %d as explored "
              "(drone in cone, angle=%.1f°, dist=%.1fm from jct)",
              jct.id, angle * 180.0 / M_PI, dist_cur);
          }
        }
      }

      jct.fully_explored = (jct.unexplored_arm_count() == 0);
      if (jct.fully_explored) {
        RCLCPP_INFO(get_logger(),
          "[JUNCTION] Junction %d fully explored", jct.id);
      }
    }
  }

  /**
   * Periodically re-evaluate the exploration gain of all junction arms
   * using the current OctoMap.
   */
  void re_evaluate_junction_gains() {
    if (!octree_) return;

    for (auto& jct : junctions_) {
      if (jct.fully_explored) continue;

      for (auto& arm : jct.arms) {
        if (arm.explored) continue;
        arm.exploration_gain = calculate_exploration_gain(arm.target);
      }

      // Update fully_explored status
      jct.fully_explored = (jct.unexplored_arm_count() == 0);
    }
  }

  /**
   * Compute exploration gain: ratio of unknown voxels within radius of a point.
   * Returns [0, 1].
   */
  float calculate_exploration_gain(const geometry_msgs::msg::Point& pos) {
    if (!octree_) return 0.0f;

    octomap::point3d center(pos.x, pos.y, pos.z);

    int unknown = 0, total = 0;
    float res = static_cast<float>(octree_->getResolution());
    float step = std::max(res, 1.0f);  // Coarse sampling for speed

    for (double dx = -exploration_eval_radius_; dx <= exploration_eval_radius_; dx += step) {
      for (double dy = -exploration_eval_radius_; dy <= exploration_eval_radius_; dy += step) {
        for (double dz = -exploration_eval_radius_; dz <= exploration_eval_radius_; dz += step) {
          octomap::point3d p = center + octomap::point3d(dx, dy, dz);
          total++;
          if (!octree_->search(p)) {
            unknown++;
          }
        }
      }
    }

    if (total == 0) return 0.0f;
    return static_cast<float>(unknown) / static_cast<float>(total);
  }

  // ==========================================================================
  // Goal Publishing
  // ==========================================================================

  /**
   * Publish a single goal on /global_goal as a PoseArray (same format the RRT expects).
   */
  void publish_global_goal(const geometry_msgs::msg::Point& target) {
    geometry_msgs::msg::PoseArray goal_msg;
    goal_msg.header.stamp = now();
    goal_msg.header.frame_id = "world";

    geometry_msgs::msg::Pose pose;
    pose.position = target;
    pose.orientation.w = 1.0;
    goal_msg.poses.push_back(pose);

    global_goal_pub_->publish(goal_msg);

    RCLCPP_DEBUG(get_logger(),
      "[GLOBAL] Published goal (%.1f, %.1f, %.1f) state=%s",
      target.x, target.y, target.z, state_to_string(state_).c_str());
  }

  // ==========================================================================
  // Visualization
  // ==========================================================================

  void publish_markers() {
    visualization_msgs::msg::MarkerArray marker_array;

    // Clear previous markers
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear);

    int id = 0;

    // Draw junctions
    for (const auto& jct : junctions_) {
      // Junction center sphere
      visualization_msgs::msg::Marker jct_marker;
      jct_marker.header.frame_id = "world";
      jct_marker.header.stamp = now();
      jct_marker.ns = "junctions";
      jct_marker.id = id++;
      jct_marker.type = visualization_msgs::msg::Marker::SPHERE;
      jct_marker.action = visualization_msgs::msg::Marker::ADD;
      jct_marker.pose.position = jct.position;
      jct_marker.pose.orientation.w = 1.0;
      jct_marker.scale.x = jct_marker.scale.y = jct_marker.scale.z = 2.0;
      jct_marker.lifetime = rclcpp::Duration(std::chrono::seconds(2));

      if (jct.fully_explored) {
        jct_marker.color = make_color(0.5, 0.5, 0.5, 0.5);  // Gray
      } else {
        jct_marker.color = make_color(0.0, 0.8, 1.0, 0.9);  // Cyan
      }
      marker_array.markers.push_back(jct_marker);

      // Junction label
      visualization_msgs::msg::Marker label;
      label.header.frame_id = "world";
      label.header.stamp = now();
      label.ns = "junction_labels";
      label.id = id++;
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.action = visualization_msgs::msg::Marker::ADD;
      label.pose.position = jct.position;
      label.pose.position.z += 2.5;
      label.pose.orientation.w = 1.0;
      label.scale.z = 1.5;
      label.color = make_color(1, 1, 1, 1);
      label.lifetime = rclcpp::Duration(std::chrono::seconds(2));

      char buf[128];
      snprintf(buf, sizeof(buf), "J%d (%d arms, %d open)",
        jct.id, static_cast<int>(jct.arms.size()), jct.unexplored_arm_count());
      label.text = buf;
      marker_array.markers.push_back(label);

      // Arms as arrows
      for (const auto& arm : jct.arms) {
        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = "world";
        arrow.header.stamp = now();
        arrow.ns = "junction_arms";
        arrow.id = id++;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        arrow.lifetime = rclcpp::Duration(std::chrono::seconds(2));

        geometry_msgs::msg::Point start_pt = jct.position;
        geometry_msgs::msg::Point end_pt;
        end_pt.x = jct.position.x + arm.direction.x() * 8.0;
        end_pt.y = jct.position.y + arm.direction.y() * 8.0;
        end_pt.z = jct.position.z + arm.direction.z() * 8.0;
        arrow.points.push_back(start_pt);
        arrow.points.push_back(end_pt);

        arrow.scale.x = 0.5;  // Shaft diameter
        arrow.scale.y = 1.0;  // Head diameter
        arrow.scale.z = 0.0;

        if (arm.explored) {
          arrow.color = make_color(0.5, 0.5, 0.5, 0.6);  // Gray = explored
        } else if (arm.exploration_gain > min_gain_threshold_) {
          arrow.color = make_color(0.0, 1.0, 0.0, 0.9);  // Green = open
        } else {
          arrow.color = make_color(1.0, 0.3, 0.0, 0.7);  // Orange = low gain
        }
        marker_array.markers.push_back(arrow);
      }
    }

    // Home position marker
    visualization_msgs::msg::Marker home;
    home.header.frame_id = "world";
    home.header.stamp = now();
    home.ns = "home";
    home.id = id++;
    home.type = visualization_msgs::msg::Marker::CUBE;
    home.action = visualization_msgs::msg::Marker::ADD;
    home.pose.position = home_position_;
    home.pose.orientation.w = 1.0;
    home.scale.x = home.scale.y = home.scale.z = 3.0;
    home.color = make_color(1.0, 0.84, 0.0, 0.8);  // Gold
    home.lifetime = rclcpp::Duration(std::chrono::seconds(2));
    marker_array.markers.push_back(home);

    // Current state text above drone
    if (odom_received_) {
      visualization_msgs::msg::Marker state_text;
      state_text.header.frame_id = "world";
      state_text.header.stamp = now();
      state_text.ns = "state";
      state_text.id = id++;
      state_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      state_text.action = visualization_msgs::msg::Marker::ADD;
      state_text.pose.position = current_position_;
      state_text.pose.position.z += 5.0;
      state_text.pose.orientation.w = 1.0;
      state_text.scale.z = 2.0;
      state_text.color = make_color(1, 1, 1, 1);
      state_text.lifetime = rclcpp::Duration(std::chrono::seconds(2));

      char buf[256];
      int total_junctions = static_cast<int>(junctions_.size());
      int explored_junctions = 0;
      for (const auto& j : junctions_) {
        if (j.fully_explored) explored_junctions++;
      }

      snprintf(buf, sizeof(buf), "[%s]\nJunctions: %d/%d explored",
        state_to_string(state_).c_str(), explored_junctions, total_junctions);
      state_text.text = buf;
      marker_array.markers.push_back(state_text);
    }

    marker_pub_->publish(marker_array);
  }

  // ==========================================================================
  // Wall Clearance
  // ==========================================================================

  /**
   * Check whether a point has at least `clearance` metres of free space
   * around it (no occupied OctoMap voxels within that radius).
   */
  bool has_wall_clearance(const geometry_msgs::msg::Point& pos, double clearance) {
    if (!octree_) return true;  // No map yet — assume OK

    float res = static_cast<float>(octree_->getResolution());
    float step = std::max(res, 0.5f);  // Fine-ish sampling
    octomap::point3d center(pos.x, pos.y, pos.z);

    for (double dx = -clearance; dx <= clearance; dx += step) {
      for (double dy = -clearance; dy <= clearance; dy += step) {
        for (double dz = -clearance; dz <= clearance; dz += step) {
          if (dx*dx + dy*dy + dz*dz > clearance * clearance) continue;
          octomap::point3d p = center + octomap::point3d(dx, dy, dz);
          auto* node = octree_->search(p);
          if (node && octree_->isNodeOccupied(*node)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  /**
   * If `target` is too close to a wall, pull it toward `safe_anchor`
   * (typically the junction centre, which the drone already passed through)
   * until it has enough clearance, or give up and return the anchor.
   */
  geometry_msgs::msg::Point adjust_for_wall_clearance(
      const geometry_msgs::msg::Point& target,
      const geometry_msgs::msg::Point& safe_anchor,
      double clearance) {

    if (has_wall_clearance(target, clearance)) {
      return target;  // Already fine
    }

    RCLCPP_WARN(get_logger(),
      "[WALL CHECK] Backtrack target (%.1f, %.1f, %.1f) is within %.1fm of a wall — adjusting",
      target.x, target.y, target.z, clearance);

    // Linearly interpolate from target toward safe_anchor in 10% steps
    for (double t = 0.1; t <= 1.0; t += 0.1) {
      geometry_msgs::msg::Point candidate;
      candidate.x = target.x + t * (safe_anchor.x - target.x);
      candidate.y = target.y + t * (safe_anchor.y - target.y);
      candidate.z = target.z + t * (safe_anchor.z - target.z);

      if (has_wall_clearance(candidate, clearance)) {
        RCLCPP_INFO(get_logger(),
          "[WALL CHECK] Adjusted target to (%.1f, %.1f, %.1f) (%.0f%% toward junction)",
          candidate.x, candidate.y, candidate.z, t * 100);
        return candidate;
      }
    }

    // Worst case: use the junction centre itself
    RCLCPP_WARN(get_logger(),
      "[WALL CHECK] Could not find clear point — falling back to junction centre (%.1f, %.1f, %.1f)",
      safe_anchor.x, safe_anchor.y, safe_anchor.z);
    return safe_anchor;
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  static double point_distance(const geometry_msgs::msg::Point& a,
                               const geometry_msgs::msg::Point& b) {
    return std::sqrt(
      std::pow(a.x - b.x, 2) +
      std::pow(a.y - b.y, 2) +
      std::pow(a.z - b.z, 2));
  }

  static std_msgs::msg::ColorRGBA make_color(float r, float g, float b, float a) {
    std_msgs::msg::ColorRGBA c;
    c.r = r; c.g = g; c.b = b; c.a = a;
    return c;
  }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
