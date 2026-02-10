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
  }
  
  void updateDesiredStateFromCarrot() {
      if (path_points_.empty()) return;

      // 1. Find closest point on path to current drone position 'x'
      double min_dist_sq = std::numeric_limits<double>::max();
      size_t closest_idx = 0;
      
      // Optimization: If path is long, we could search only a window, but full search is safest
      for (size_t i = 0; i < path_points_.size(); ++i) {
          const auto& p = path_points_[i];
          double dx = p.transforms[0].translation.x - x.x();
          double dy = p.transforms[0].translation.y - x.y();
          double dz = p.transforms[0].translation.z - x.z();
          double d_sq = dx*dx + dy*dy + dz*dz;
           if (d_sq < min_dist_sq) {
               min_dist_sq = d_sq;
               closest_idx = i;
           }
      }

      // 2. Look ahead 5m (or end of path)
      size_t target_idx = closest_idx;
      double lookahead = 5.0; 
      
      for (size_t i = closest_idx; i < path_points_.size(); ++i) {
          const auto& p = path_points_[i];
          double dx = p.transforms[0].translation.x - x.x();
          double dy = p.transforms[0].translation.y - x.y();
          double dz = p.transforms[0].translation.z - x.z();
          double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
          
          target_idx = i;
          if (dist >= lookahead) {
              break;
          }
      }
      
      const auto& target = path_points_[target_idx];

      // 3. Update Desired State
      if (!target.transforms.empty()) {
            xd << target.transforms[0].translation.x,
                    target.transforms[0].translation.y,
                    target.transforms[0].translation.z;
            
            // Extract yaw
            auto q = target.transforms[0].rotation;
            double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
            double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            yawd = std::atan2(siny_cosp, cosy_cosp);
      }

      if (!target.velocities.empty()) {
            vd << target.velocities[0].linear.x,
                    target.velocities[0].linear.y,
                    target.velocities[0].linear.z;
      }

      if (!target.accelerations.empty()) {
            ad << target.accelerations[0].linear.x,
                    target.accelerations[0].linear.y,
                    target.accelerations[0].linear.z;
      }
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
    updateDesiredStateFromCarrot();

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

     static int counter = 0;
     counter++;

     if (counter % 500 == 0) {  // bei 1000 Hz => etwa alle 0.5s
         double pos_error = ex.norm();
         RCLCPP_INFO(this->get_logger(), "Position error: %.3f m", pos_error);
     }
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControllerNode>());
  rclcpp::shutdown();
  return 0;
}
