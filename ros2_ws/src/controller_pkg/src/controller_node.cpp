#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>

#include <mav_msgs/msg/actuators.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Dense>
#include <tf2/utils.h>

#define PI M_PI


class ControllerNode : public rclcpp::Node {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  PART 1 |  Declare ROS callback handlers

  // Subscribers
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr desired_state_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr current_state_subscriber_;

  // Publisher (propeller speeds)
  rclcpp::Publisher<mav_msgs::msg::Actuators>::SharedPtr rotor_speed_publisher_;

  // Debug visualization publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_publisher_;

  // Timer for main control loop
  rclcpp::TimerBase::SharedPtr control_loop_timer_;


  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                                 end part 1
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Controller parameters
  double kx, kv, kr, komega; // controller gains - [1] eq (15), (16)

  // Physical constants (we will set them below)
  double m;              // mass of the UAV
  double g;              // gravity acceleration
  double d;              // distance from the center of propellers to the c.o.m.
  double cf,             // Propeller lift coefficient
         cd;             // Propeller drag coefficient
  Eigen::Matrix3d J;     // Inertia Matrix
  Eigen::Vector3d e3;    // [0,0,1]
  Eigen::MatrixXd F2W;   // Wrench-rotor speeds map

  // Controller internals (you will have to set them below)
  // Current state
  Eigen::Vector3d x;     // current position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d v;     // current velocity of the UAV's c.o.m. in the world frame
  Eigen::Matrix3d R;     // current orientation of the UAV
  Eigen::Vector3d omega; // current angular velocity of the UAV's c.o.m. in the *body* frame

  // Desired state
  Eigen::Vector3d xd;    // desired position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d vd;    // desired velocity of the UAV's c.o.m. in the world frame
  Eigen::Vector3d ad;    // desired acceleration of the UAV's c.o.m. in the world frame
  double yawd;           // desired yaw angle
  
  // Storage for the full trajectory path (Carrot Following)
  std::vector<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint> path_points_;
  bool has_trajectory_ = false;  // true once first trajectory is received

  // Debug state (updated by updateDesiredStateFromCarrot, published in controlLoop)
  double debug_lookahead_ = 0.0;
  double debug_desired_speed_ = 0.0;
  double debug_dist_to_end_ = 0.0;
  double debug_cross_track_ = 0.0;
  Eigen::Vector3d debug_path_tangent_ = Eigen::Vector3d::Zero();
  int debug_publish_counter_ = 0;

  double hz;             // frequency of the main control loop

  static Eigen::Vector3d Vee(const Eigen::Matrix3d& in){
    Eigen::Vector3d out;
    out << in(2,1), in(0,2), in(1,0);
    return out;
  }

  static double signed_sqrt(double val){
    return val>0?sqrt(val):-sqrt(-val);
  }

public:
  ControllerNode()
  : rclcpp::Node("controller_node"),
    e3(0,0,1),
    F2W(4,4),
    hz(1000.0),
    // Initial condition: 3m above where the drone spawns
    xd(-38,10,9),
    vd(0,0,0),
    ad(0,0,0),
    yawd(0)
  {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 2 |  Initialize ROS callback handlers
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    // bind controllerNode::onDesiredState() to the topic "desired_state"
    desired_state_subscriber_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
        "/command/trajectory", 10,
        std::bind(&ControllerNode::onDesiredState, this, std::placeholders::_1));

    // bind controllerNode::onCurrentState() to the topic "current_state"
    current_state_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/current_state_est", 10,
        std::bind(&ControllerNode::onCurrentState, this, std::placeholders::_1));

    // Publisher for Motor Commands
    rotor_speed_publisher_ = this->create_publisher<mav_msgs::msg::Actuators>(
        "/rotor_speed_cmds", 10);

    // Debug visualization publisher
    debug_marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/controller_debug", 10);

    // bind controllerNode::controlLoop() to the created timer, at frequency
    control_loop_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / hz),
        std::bind(&ControllerNode::controlLoop, this));

    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //                                 end part 2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 6 [NOTE: save this for last] |  Tune your gains!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Controller gains 
    kx = 3.0;
    kv = 3.6; 
    kr = 10.0;
    komega = 2.0;

    // Initialize constants
    m = 1.0;
    cd = 1e-5;
    cf = 1e-3;
    g = 9.81;
    d = 0.3;
    J << 1.0,0.0,0.0,
         0.0,1.0,0.0,
         0.0,0.0,1.0;

    RCLCPP_INFO(this->get_logger(), "controller_node ready (hz=%.1f)", hz);
  }

  void onDesiredState(const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg){
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  PART 3 | Objective: fill in xd, vd, ad, yawd
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      if (msg->points.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Received empty trajectory message");
        return;
      }

      // Store the full path for the control loop to process (Carrot Following)
      path_points_ = msg->points;
      has_trajectory_ = true;
  }
  
  // Helper: extract position from a trajectory point
  Eigen::Vector3d pointPosition(const trajectory_msgs::msg::MultiDOFJointTrajectoryPoint& pt) const {
      return Eigen::Vector3d(
          pt.transforms[0].translation.x,
          pt.transforms[0].translation.y,
          pt.transforms[0].translation.z);
  }

  // Helper: shortest distance from point p to line segment (a, b)
  static double pointToSegmentDist(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& a,
                                   const Eigen::Vector3d& b) {
      Eigen::Vector3d ab = b - a;
      double len_sq = ab.squaredNorm();
      if (len_sq < 1e-12) return (p - a).norm();
      double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
      return (p - (a + t * ab)).norm();
  }

  void updateDesiredStateFromCarrot() {
      if (path_points_.empty()) return;

      // ── Multi-point trajectory (RRT paths) → Pure Pursuit ──
      double min_dist_sq = std::numeric_limits<double>::max();
      size_t closest_idx = 0;

      for (size_t i = 0; i < path_points_.size(); ++i) {
          Eigen::Vector3d p = pointPosition(path_points_[i]);
          double d_sq = (p - x).squaredNorm();
          if (d_sq < min_dist_sq) {
              min_dist_sq = d_sq;
              closest_idx = i;
          }
      }

      // Prune waypoints we've clearly passed (keep a small margin)
      if (closest_idx > 1) {
          path_points_.erase(path_points_.begin(),
                             path_points_.begin() + static_cast<long>(closest_idx - 1));
          closest_idx = 1;
      }

      // ── 2. Cross-track error: distance to nearest path *segment* ──
      //    (not just nearest waypoint — waypoints can be 4m+ apart)
      double cross_track_error = std::sqrt(min_dist_sq); // fallback
      for (size_t i = 0; i + 1 < path_points_.size(); ++i) {
          double d = pointToSegmentDist(x, pointPosition(path_points_[i]),
                                           pointPosition(path_points_[i + 1]));
          cross_track_error = std::min(cross_track_error, d);
      }

      // ── 3. Adaptive lookahead: scale with current speed ──
      double speed = v.norm();
      double lookahead_min = 3.0;   // metres
      double lookahead_max = 10.0;
      double lookahead = std::clamp(speed * 1.2, lookahead_min, lookahead_max);

      // If significantly off-path, shrink lookahead to pull back onto path
      if (cross_track_error > 3.0) {
          lookahead = lookahead_min;
      }

      // ── 4. Walk along path segments from closest point, accumulate
      //       arc-length until we exceed the lookahead distance.
      //       Then *interpolate* the exact carrot point. ──
      Eigen::Vector3d carrot = pointPosition(path_points_.back()); // fallback: end of path
      Eigen::Vector3d path_tangent = Eigen::Vector3d::Zero();
      bool interpolated = false;

      double arc = 0.0;
      for (size_t i = closest_idx; i + 1 < path_points_.size(); ++i) {
          Eigen::Vector3d seg_start = pointPosition(path_points_[i]);
          Eigen::Vector3d seg_end   = pointPosition(path_points_[i + 1]);
          Eigen::Vector3d seg       = seg_end - seg_start;
          double seg_len = seg.norm();

          if (seg_len < 1e-6) continue;

          if (arc + seg_len >= lookahead) {
              // Carrot lies on this segment — interpolate
              double t = (lookahead - arc) / seg_len;
              carrot = seg_start + t * seg;
              path_tangent = seg.normalized();
              interpolated = true;
              break;
          }
          arc += seg_len;
      }

      // If we reached the end of path without exceeding lookahead,
      // tangent points from second-to-last → last waypoint
      if (!interpolated && path_points_.size() >= 2) {
          Eigen::Vector3d p0 = pointPosition(path_points_[path_points_.size() - 2]);
          Eigen::Vector3d p1 = pointPosition(path_points_.back());
          Eigen::Vector3d seg = p1 - p0;
          path_tangent = (seg.norm() > 1e-6) ? seg.normalized() : Eigen::Vector3d::Zero();
      }

      // ── 5. Set desired position ──
      xd = carrot;

      // ── 6. Compute remaining path distance ──
      double dist_to_end = 0.0;
      for (size_t i = closest_idx; i + 1 < path_points_.size(); ++i) {
          dist_to_end += (pointPosition(path_points_[i + 1])
                        - pointPosition(path_points_[i])).norm();
      }

      // ── 7. Velocity feedforward along path tangent ──
      double dist_to_carrot = (carrot - x).norm();
      double desired_speed = std::min(dist_to_carrot * 1.2, 7.0);  // soft speed limit 7 m/s

      // Kinematic braking near end of path: v = sqrt(2*a*d), a=4 m/s²
      double brake_speed = std::sqrt(2.0 * 4.0 * dist_to_end);
      desired_speed = std::min(desired_speed, brake_speed);

      // Cross-track speed reduction: if drone is far from path, slow down
      // Uses proper segment distance so this only triggers when truly off-path
      if (cross_track_error > 2.0) {
          double cross_factor = 2.0 / cross_track_error;  // e.g. 4m off → 50% speed
          desired_speed *= cross_factor;
      }

      // If remaining path is very short, just hover (prevents drifting into wall)
      if (dist_to_end < 2.0) {
          desired_speed = 0.0;
      }

      if (path_tangent.norm() > 0.5 && desired_speed > 0.01) {
          vd = path_tangent * desired_speed;
      } else {
          vd.setZero();
      }

      // No acceleration feedforward from RRT paths
      ad.setZero();

      // Store debug values for visualization
      debug_lookahead_ = lookahead;
      debug_desired_speed_ = desired_speed;
      debug_dist_to_end_ = dist_to_end;
      debug_cross_track_ = cross_track_error;
      debug_path_tangent_ = path_tangent;

      // ── 8. Yaw: face direction of travel (path tangent in xy-plane) ──
      if (path_tangent.head<2>().norm() > 0.1) {
          yawd = std::atan2(path_tangent.y(), path_tangent.x());
      }
      // else: keep previous yawd (avoid spinning when moving vertically)
  }


  void onCurrentState(const nav_msgs::msg::Odometry::SharedPtr cur_state_msg){
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  PART 4 | Objective: fill in x, v, R and omega
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      // 4.1 Position and linear velocity
      x << cur_state_msg->pose.pose.position.x,
          cur_state_msg->pose.pose.position.y,
          cur_state_msg->pose.pose.position.z;

      v << cur_state_msg->twist.twist.linear.x,
          cur_state_msg->twist.twist.linear.y,
          cur_state_msg->twist.twist.linear.z;

      // 4.2 Orientation (convert quaternion to rotation matrix)
      Eigen::Quaterniond q(
          cur_state_msg->pose.pose.orientation.w,
          cur_state_msg->pose.pose.orientation.x,
          cur_state_msg->pose.pose.orientation.y,
          cur_state_msg->pose.pose.orientation.z);
      R = q.toRotationMatrix();

      // 4.3 Angular velocity (convert from world frame to body frame)
      Eigen::Vector3d omega_world;
      omega_world << cur_state_msg->twist.twist.angular.x,
                      cur_state_msg->twist.twist.angular.y,
                      cur_state_msg->twist.twist.angular.z;
      omega = R.transpose() * omega_world;
  }

  void controlLoop(){
    // Before first trajectory: hover at current position
    if (!has_trajectory_) {
        xd = x;
        vd.setZero();
        ad.setZero();
    } else {
        updateDesiredStateFromCarrot();
    }

    Eigen::Vector3d ex, ev, er, eomega;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 5 | Objective: Implement the controller!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // 5.1 Compute position and velocity errors. Objective: fill in ex, ev.
    ex = x - xd;
    ev = v - vd;

    // Limit Max Velocity (by saturation of position error)
    // Limits the maximum "pull" towards the goal
    // Steady state velocity v = -(kx/kv) * ex
    double max_velocity = 8.0; // m/s
    double max_ex = max_velocity * kv / kx;
    if (ex.norm() > max_ex) {
        ex = ex.normalized() * max_ex;
    }

    // 5.2 Compute the Rd matrix.

    // Desired body z-axis (force direction)
    Eigen::Vector3d a_fb = (-kx * ex - kv * ev) / m;

    // Limit Max Acceleration
    double max_acceleration = 5.0; // m/s^2
    if (a_fb.norm() > max_acceleration) {
        a_fb = a_fb.normalized() * max_acceleration;
    }

    Eigen::Vector3d force_dir = m * a_fb + m * g * e3 + m * ad;
    Eigen::Vector3d b3d = force_dir.normalized();

    // Desired yaw direction (projection on xy-plane)
    Eigen::Vector3d b1d(cos(yawd), sin(yawd), 0.0);

    // Compute orthogonal axes
    Eigen::Vector3d b2d = (b3d.cross(b1d)).normalized();
    Eigen::Vector3d b1c = (b2d.cross(b3d)).normalized();

    // Assemble rotation matrix
    Eigen::Matrix3d Rd;
    Rd << b1c, b2d, b3d;

    //
    // 5.3 Compute the orientation error (er) and the rotation-rate error (eomega)

    Eigen::Matrix3d R_d_T_R = Rd.transpose() * R;
    er = 0.5 * Vee(R_d_T_R - R_d_T_R.transpose());
    eomega = omega;  // since omega_d ≈ 0


    // 5.4 Compute the desired wrench (force + torques) to control the UAV.
    //  Hints:
    //     - [1] eq. (15), (16)
    //

    double f = (-kx * ex - kv * ev + m * g * e3 + m * ad).dot(R * e3);
    Eigen::Vector3d M = -kr * er - komega * eomega + omega.cross(J*omega);

    // 5.5 Recover the rotor speeds from the wrench computed above
    Eigen::Vector4d wrench;
    wrench << f, M.x(), M.y(), M.z();

    // Allocation matrix (Wrench -> rotor speeds^2)
    Eigen::Matrix4d W2F;
    double a = cf * d / std::sqrt(2.0);

    // f, τx, τy, τz = W2F * [w1², w2², w3², w4²]
    W2F <<
        //  w1       w2       w3       w4
        cf,        cf,       cf,      cf,       // total thrust
        +a,        +a,      -a,     -a,          // roll  (τx)
        -a,        +a,      +a,     -a,          // pitch (τy)
        cd,      -cd,      +cd,    -cd;         // yaw   (τz)

    F2W = W2F.inverse();


    Eigen::Vector4d w_sq = F2W * wrench;

    // Take signed square root
    double w1 = signed_sqrt(w_sq(0));
    double w2 = signed_sqrt(w_sq(1));
    double w3 = signed_sqrt(w_sq(2));
    double w4 = signed_sqrt(w_sq(3));

    //
    // ~~~~ end solution
    //
    // 5.6 Populate and publish the control message

    mav_msgs::msg::Actuators cmd;
    cmd.angular_velocities.resize(4);
    cmd.angular_velocities[0] = w1;
    cmd.angular_velocities[1] = w2;
    cmd.angular_velocities[2] = w3;
    cmd.angular_velocities[3] = w4;
    rotor_speed_publisher_->publish(cmd);

    // Publish debug markers at ~20 Hz (every 50th iteration at 1000 Hz)
    debug_publish_counter_++;
    if (debug_publish_counter_ >= 50) {
        debug_publish_counter_ = 0;
        publishDebugMarkers(ex, ev);
    }
  }

  void publishDebugMarkers(const Eigen::Vector3d& ex, const Eigen::Vector3d& ev) {
      auto now = this->get_clock()->now();
      visualization_msgs::msg::MarkerArray markers;
      int id = 0;

      // ── 1. Carrot point (cyan sphere) ──
      {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_carrot";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::SPHERE;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = xd.x();
          m.pose.position.y = xd.y();
          m.pose.position.z = xd.z();
          m.pose.orientation.w = 1.0;
          m.scale.x = 0.6; m.scale.y = 0.6; m.scale.z = 0.6;
          m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 0.9f;
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          markers.markers.push_back(m);
      }

      // ── 2. Line from drone to carrot (cyan) ──
      {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_line_to_carrot";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::LINE_STRIP;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.orientation.w = 1.0;
          m.scale.x = 0.08;
          m.color.r = 0.0f; m.color.g = 0.8f; m.color.b = 1.0f; m.color.a = 0.6f;
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          geometry_msgs::msg::Point p1, p2;
          p1.x = x.x(); p1.y = x.y(); p1.z = x.z();
          p2.x = xd.x(); p2.y = xd.y(); p2.z = xd.z();
          m.points.push_back(p1);
          m.points.push_back(p2);
          markers.markers.push_back(m);
      }

      // ── 3. Current velocity arrow (green) ──
      {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_vel_actual";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::ARROW;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.orientation.w = 1.0;
          m.scale.x = 0.1; m.scale.y = 0.2; m.scale.z = 0.0;
          m.color.r = 0.2f; m.color.g = 1.0f; m.color.b = 0.2f; m.color.a = 0.9f;
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          geometry_msgs::msg::Point start, end;
          start.x = x.x(); start.y = x.y(); start.z = x.z();
          end.x = x.x() + v.x(); end.y = x.y() + v.y(); end.z = x.z() + v.z();
          m.points.push_back(start);
          m.points.push_back(end);
          markers.markers.push_back(m);
      }

      // ── 4. Desired velocity arrow (yellow) ──
      {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_vel_desired";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::ARROW;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.orientation.w = 1.0;
          m.scale.x = 0.1; m.scale.y = 0.2; m.scale.z = 0.0;
          m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 0.9f;
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          geometry_msgs::msg::Point start, end;
          start.x = x.x(); start.y = x.y(); start.z = x.z();
          end.x = x.x() + vd.x(); end.y = x.y() + vd.y(); end.z = x.z() + vd.z();
          m.points.push_back(start);
          m.points.push_back(end);
          markers.markers.push_back(m);
      }

      // ── 5. Text overlay near drone with debug info ──
      {
          char buf[320];
          std::snprintf(buf, sizeof(buf),
              "v=%.1f / vd=%.1f m/s\n"
              "pos_err=%.2f m\n"
              "cross_trk=%.2f m\n"
              "lookahead=%.1f m\n"
              "dist_end=%.1f m\n"
              "pts=%zu",
              v.norm(), vd.norm(),
              ex.norm(),
              debug_cross_track_,
              debug_lookahead_,
              debug_dist_to_end_,
              path_points_.size());

          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_text";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = x.x();
          m.pose.position.y = x.y();
          m.pose.position.z = x.z() + 1.5; // above drone
          m.pose.orientation.w = 1.0;
          m.scale.z = 0.5; // text height
          m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 1.0f;
          m.text = std::string(buf);
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          markers.markers.push_back(m);
      }

      // ── 6. Remaining path (thin white line) ──
      if (path_points_.size() >= 2) {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "world";
          m.header.stamp = now;
          m.ns = "ctrl_remaining_path";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::LINE_STRIP;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.orientation.w = 1.0;
          m.scale.x = 0.05;
          m.color.r = 0.8f; m.color.g = 0.8f; m.color.b = 0.8f; m.color.a = 0.5f;
          m.lifetime = rclcpp::Duration::from_seconds(0.2);
          for (const auto& pt : path_points_) {
              geometry_msgs::msg::Point p;
              p.x = pt.transforms[0].translation.x;
              p.y = pt.transforms[0].translation.y;
              p.z = pt.transforms[0].translation.z;
              m.points.push_back(p);
          }
          markers.markers.push_back(m);
      }

      debug_marker_publisher_->publish(markers);
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControllerNode>());
  rclcpp::shutdown();
  return 0;
}
