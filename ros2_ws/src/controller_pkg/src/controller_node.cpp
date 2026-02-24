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

  // Publisher for target point visualization
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_publisher_;

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
  
  // Storage for the full trajectory path (Time-based trajectory following)
  std::vector<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint> path_points_;
  rclcpp::Time trajectory_start_time_;  // Timestamp when current trajectory was received

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
    yawd(3.1415926)
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

    // Publisher for target point visualization
    target_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/target_point", 10);

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

            // If message contains only a single point that's almost exactly
            // the current drone position, ignore it so we keep following the
            // previously received trajectory. This avoids overwriting the
            // path with a trivial start-point when planning fails.
            bool accept_new = true;
            if (msg->points.size() == 1 && !path_points_.empty()) {
                const auto &p0 = msg->points[0];
                if (!p0.transforms.empty()) {
                    double px = p0.transforms[0].translation.x;
                    double py = p0.transforms[0].translation.y;
                    double pz = p0.transforms[0].translation.z;
                    double dx = px - x.x();
                    double dy = py - x.y();
                    double dz = pz - x.z();
                    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist < 0.2) {
                        RCLCPP_INFO(this->get_logger(), "Ignoring trivial single-point trajectory (dist=%.3f)", dist);
                        accept_new = false;
                    }
                }
            }

            if (accept_new) {
                // Store the full path and record when trajectory was received
                path_points_ = msg->points;
                trajectory_start_time_ = this->get_clock()->now();
            }
  }
  
  void updateDesiredStateFromTime() {
      if (path_points_.empty()) return;

      // Calculate elapsed time since trajectory was received
      double elapsed_time = (this->get_clock()->now() - trajectory_start_time_).seconds();

      // Start search at index 1 if available to ignore the initial 'current position' waypoint
      size_t start_idx = (path_points_.size() > 1) ? 1 : 0;

      // Find trajectory point that matches current elapsed time
      // Strategy: Find first point where time_from_start >= elapsed_time, starting at start_idx
      size_t target_idx = start_idx;
      for (size_t i = start_idx; i < path_points_.size(); ++i) {
          const auto& p = path_points_[i];
          double point_time = p.time_from_start.sec + p.time_from_start.nanosec / 1e9;

          if (point_time >= elapsed_time) {
              target_idx = i;
              break;
          }
      }

      // If no point matched the elapsed_time, use the last point
      target_idx = std::min(target_idx, path_points_.size() - 1);

      // If the chosen target is still too close to the drone (e.g. first point equals drone position),
      // advance to the next waypoint so the controller doesn't 'stick' to the drone's current pose.
      const auto& tentative = path_points_[target_idx];
      double ignore_first_threshold = 0.5; // meters
      if (!tentative.transforms.empty()) {
          double tx = tentative.transforms[0].translation.x;
          double ty = tentative.transforms[0].translation.y;
          double tz = tentative.transforms[0].translation.z;
          double dx_now = tx - x.x();
          double dy_now = ty - x.y();
          double dz_now = tz - x.z();
          double dist_now = std::sqrt(dx_now*dx_now + dy_now*dy_now + dz_now*dz_now);

          if (dist_now < ignore_first_threshold && target_idx + 1 < path_points_.size()) {
              target_idx = target_idx + 1;
          }
      }

      const auto& target = path_points_[target_idx];

      // Update Desired State from target trajectory point
      if (!target.transforms.empty()) {
            xd << target.transforms[0].translation.x,
                    target.transforms[0].translation.y,
                    target.transforms[0].translation.z;

            // Extract yaw from quaternion
            auto q = target.transforms[0].rotation;
            double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
            double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            yawd = std::atan2(siny_cosp, cosy_cosp);
      }

      // Calculate desired velocity from consecutive trajectory points
      if (target_idx + 1 < path_points_.size()) {
          const auto& next_target = path_points_[target_idx + 1];

          if (!target.transforms.empty() && !next_target.transforms.empty()) {
              // Get position difference
              double dx = next_target.transforms[0].translation.x - target.transforms[0].translation.x;
              double dy = next_target.transforms[0].translation.y - target.transforms[0].translation.y;
              double dz = next_target.transforms[0].translation.z - target.transforms[0].translation.z;

              // Get time difference
              double curr_time = target.time_from_start.sec + target.time_from_start.nanosec / 1e9;
              double next_time = next_target.time_from_start.sec + next_target.time_from_start.nanosec / 1e9;
              double dt = next_time - curr_time;

              // Calculate desired velocity (position delta / time delta)
              if (dt > 1e-6) {
                  vd << dx / dt, dy / dt, dz / dt;
              } else {
                  vd.setZero();
              }
          } else {
              vd.setZero();
          }
      } else {
          // Last point: no more movement desired
          vd.setZero();
      }

      // No acceleration data from trajectory (all zeros anyway)
      ad.setZero();
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
    updateDesiredStateFromTime();

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
    double max_velocity = 10.0; // m/s
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

    // 5.7 Publish target point marker for visualization
    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = "world";
    target_marker.header.stamp = this->get_clock()->now();
    target_marker.id = 0;
    target_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_marker.action = visualization_msgs::msg::Marker::ADD;
    
    target_marker.pose.position.x = xd.x();
    target_marker.pose.position.y = xd.y();
    target_marker.pose.position.z = xd.z();
    target_marker.pose.orientation.w = 1.0;
    
    target_marker.scale.x = 1.0;
    target_marker.scale.y = 1.0;
    target_marker.scale.z = 1.0;
    
    // Cyan color with some transparency
    target_marker.color.r = 0.0f;
    target_marker.color.g = 1.0f;
    target_marker.color.b = 1.0f;
    target_marker.color.a = 0.8f;
    
    target_marker_publisher_->publish(target_marker);

     static int counter = 0;
     counter++;

     if (counter % 500 == 0) {  // bei 1000 Hz => etwa alle 0.5s
         double pos_error = ex.norm();
         //RCLCPP_INFO(this->get_logger(), "Position error: %.3f m", pos_error);
     }
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControllerNode>());
  rclcpp::shutdown();
  return 0;
}
