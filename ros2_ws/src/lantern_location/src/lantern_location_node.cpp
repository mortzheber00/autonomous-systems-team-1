// Standard library includes
#include <memory>
#include <string>
#include <cstring>

// ROS2 includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Define the namespace for this node
namespace lantern_location
{

// Sync policy for approximate time synchronization
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
  sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

// Structure to track detected lanterns
struct DetectedLantern
{
  geometry_msgs::msg::Point position;  // Average position in world frame
  geometry_msgs::msg::Point sum;       // Sum of all detections for true average
  int update_count;                     // Number of times this lantern was updated
};

class LanternLocationNode : public rclcpp::Node
{
public:
  LanternLocationNode()
  : rclcpp::Node("lantern_location_node"),
    tf_buffer_(std::make_shared<rclcpp::Clock>(RCL_ROS_TIME)),
    tf_listener_(tf_buffer_)
  {
    RCLCPP_INFO(get_logger(), "Initializing Lantern Location Node...");
    
    // Depth scale parameter (conversion from depth units to meters)
    this->declare_parameter("depth_scale", 0.001);  // mm to meters
    depth_scale_ = this->get_parameter("depth_scale").as_double();

    // Processing frame skip parameter (process every Nth frame)
    this->declare_parameter("frame_skip", 5);  // Process every 5th frame by default
    frame_skip_ = this->get_parameter("frame_skip").as_int();

    // Lantern matching distance threshold (meters in world coordinates)
    this->declare_parameter("matching_distance_threshold", 20.0);
    matching_distance_threshold_ = this->get_parameter("matching_distance_threshold").as_double();

    // Minimum updates before lantern is confirmed
    this->declare_parameter("min_updates_to_confirm", 3);
    min_updates_to_confirm_ = this->get_parameter("min_updates_to_confirm").as_int();

    // Subscribe to depth camera info to get intrinsics
    depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/realsense/depth/camera_info", 10,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!have_camera_info_) {
          // K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
          focal_length_x_ = msg->k[0];
          focal_length_y_ = msg->k[4];
          principal_point_x_ = msg->k[2];
          principal_point_y_ = msg->k[5];
          have_camera_info_ = true;
        }
      });

    // Set up synchronized subscribers using message_filters
    semantic_sub_.subscribe(this, "/Quadrotor/Sensors/SemanticCamera/image_raw",
      rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, "/realsense/depth/image",
      rmw_qos_profile_sensor_data);

    // Create synchronizer with approximate time policy (allows ~100ms time difference)
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), semantic_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&LanternLocationNode::synchronized_callback,
      this, std::placeholders::_1, std::placeholders::_2));

    // Create publishers
    filtered_depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/lantern_location/filtered_depth", 10);
    pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lantern_location/filtered_pointcloud", 10);
    lantern_locations_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/lantern_location/detected_lanterns", 10);

    // Pre-allocate buffers (will be resized on first use if needed)
    filtered_depth_ = std::make_shared<sensor_msgs::msg::Image>();
    cloud_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
    
    RCLCPP_INFO(get_logger(), "Lantern Location Node initialization complete!");
  }

private:
  // Synchronized callback - called when both images arrive with similar timestamps
  void synchronized_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& semantic_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
  {
    // Wait for camera intrinsics before processing
    if (!have_camera_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Waiting for camera info on /realsense/depth/camera_info...");
      return;
    }

    // Frame skipping for computational efficiency
    frame_counter_++;
    if (frame_counter_ % frame_skip_ != 0) {
      return;
    }

    // Check dimension match
    if (semantic_msg->width != depth_msg->width || 
        semantic_msg->height != depth_msg->height) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Image dimensions mismatch: semantic (%ux%u) vs depth (%ux%u)",
        semantic_msg->width, semantic_msg->height, depth_msg->width, depth_msg->height);
      return;
    }

    const size_t width = depth_msg->width;
    const size_t height = depth_msg->height;
    const size_t num_pixels = width * height;

    // Prepare filtered depth image (reuse buffer)
    if (filtered_depth_->data.size() != depth_msg->data.size()) {
      filtered_depth_->data.resize(depth_msg->data.size());
    }
    filtered_depth_->header = depth_msg->header;
    filtered_depth_->height = height;
    filtered_depth_->width = width;
    filtered_depth_->encoding = depth_msg->encoding;
    filtered_depth_->is_bigendian = depth_msg->is_bigendian;
    filtered_depth_->step = depth_msg->step;

    // Initialize filtered depth to zero
    std::memset(filtered_depth_->data.data(), 0, filtered_depth_->data.size());

    // Filter depth based on semantic mask
    const uint8_t* semantic_data = semantic_msg->data.data();
    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_msg->data.data());
    uint16_t* filtered_data = reinterpret_cast<uint16_t*>(filtered_depth_->data.data());

    for (size_t i = 0; i < num_pixels; ++i) {
      // Check if semantic pixel is non-zero (object of interest)
      // For rgb8: check if any channel is non-zero
      const size_t sem_idx = i * 3;
      if (semantic_data[sem_idx] != 0 || 
          semantic_data[sem_idx + 1] != 0 || 
          semantic_data[sem_idx + 2] != 0) {
        filtered_data[i] = depth_data[i];
      }
    }

    filtered_depth_pub_->publish(*filtered_depth_);

    // Build point cloud and calculate lantern location
    calculate_and_publish_lantern_location(filtered_depth_, width, height, 
                                           depth_msg->header.frame_id, 
                                           depth_msg->header.stamp);
  }

  // Transform point from camera frame to world frame using TF2
  geometry_msgs::msg::Point transform_to_world(
    const geometry_msgs::msg::Point& point_camera,
    const std::string& camera_frame,
    const rclcpp::Time& timestamp)
  {
    try {
      // Create a PointStamped in camera frame
      geometry_msgs::msg::PointStamped point_camera_stamped;
      point_camera_stamped.header.frame_id = camera_frame;
      point_camera_stamped.header.stamp = timestamp;
      point_camera_stamped.point = point_camera;

      // Transform to world frame using TF2
      geometry_msgs::msg::PointStamped point_world_stamped;
      point_world_stamped = tf_buffer_.transform(point_camera_stamped, "world", 
                                                  tf2::durationFromSec(0.1));

      return point_world_stamped.point;
    }
    catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Could not transform point: %s", ex.what());
      return point_camera;  // Return original if transform fails
    }
  }

  // Helper to setup point cloud fields (x, y, z as floats)
  void setup_pointcloud_fields(sensor_msgs::msg::PointCloud2& cloud,
                               size_t width, size_t height,
                               const std::string& frame_id)
  {
    cloud.header.frame_id = frame_id;
    cloud.height = height;
    cloud.width = width;
    cloud.is_dense = false;
    cloud.is_bigendian = false;
    cloud.point_step = 12;  // 3 floats * 4 bytes
    cloud.row_step = cloud.point_step * width;

    cloud.fields.resize(3);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;

    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;

    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;

    cloud.data.resize(cloud.point_step * width * height);
  }

  void calculate_and_publish_lantern_location(
      const sensor_msgs::msg::Image::SharedPtr& depth_image,
      size_t width, size_t height,
      const std::string& frame_id,
      const rclcpp::Time& timestamp)
  {
    const size_t num_pixels = width * height;
    const uint16_t* depth_ptr = reinterpret_cast<const uint16_t*>(depth_image->data.data());

    // Accumulate all valid points to calculate average position
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int valid_points = 0;

    // Setup point cloud structure (only reallocate if size changed)
    // Publish point cloud in world frame to match lantern markers
    if (cloud_->width != width || cloud_->height != height) {
      setup_pointcloud_fields(*cloud_, width, height, "world");
    }
    cloud_->header.stamp = timestamp;
    cloud_->header.frame_id = "world";

    // Compute 3D points in camera frame, transform to world, and accumulate for averaging
    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_, "z");

    for (size_t i = 0; i < num_pixels; ++i, ++iter_x, ++iter_y, ++iter_z) {
      const uint16_t depth_value = depth_ptr[i];

      if (depth_value == 0) {
        *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
      } else {
        // Convert pixel to 3D point in camera frame
        const int u = i % width;
        const int v = i / width;
        const float z = depth_value * depth_scale_;
        const float x = (u - principal_point_x_) * z / focal_length_x_;
        const float y = (v - principal_point_y_) * z / focal_length_y_;

        // Transform point from camera to world frame
        geometry_msgs::msg::Point point_camera;
        point_camera.x = x;
        point_camera.y = y;
        point_camera.z = z;
        geometry_msgs::msg::Point point_world = transform_to_world(point_camera, frame_id, timestamp);

        // Store in point cloud (now in world frame)
        *iter_x = point_world.x;
        *iter_y = point_world.y;
        *iter_z = point_world.z;

        // Accumulate for average calculation (in camera frame for consistency)
        sum_x += x;
        sum_y += y;
        sum_z += z;
        ++valid_points;
      }
    }

    // Publish point cloud
    pointcloud_pub_->publish(*cloud_);

    // Calculate and publish lantern location if we have valid points
    if (valid_points > 0) {
      geometry_msgs::msg::Point avg_position_camera;
      avg_position_camera.x = sum_x / valid_points;
      avg_position_camera.y = sum_y / valid_points;
      avg_position_camera.z = sum_z / valid_points;

      // Transform to world coordinates using TF2
      geometry_msgs::msg::Point lantern_world = transform_to_world(avg_position_camera, frame_id, timestamp);

      // Update or add lantern to tracking list (in world coordinates)
      update_lantern_list(lantern_world, timestamp);
    }
    
    // Clean up unconfirmed lanterns periodically (every 30 frames)
    cleanup_counter_++;
    if (cleanup_counter_ >= 30) {
      cleanup_unconfirmed_lanterns();
      cleanup_counter_ = 0;
    }
    
    // Always publish all confirmed lanterns (even when not currently detected)
    publish_lantern_locations(timestamp);
  }

  // Update or add a lantern to the tracking list (in world coordinates)
  void update_lantern_list(const geometry_msgs::msg::Point& new_position_world,
                          const rclcpp::Time& timestamp)
  {
    // Check if this lantern matches any existing lantern (within threshold distance in world frame)
    int closest_idx = -1;
    double min_distance = matching_distance_threshold_;

    for (size_t i = 0; i < detected_lanterns_.size(); ++i) {
      const auto& lantern = detected_lanterns_[i];
      double dx = new_position_world.x - lantern.position.x;
      double dy = new_position_world.y - lantern.position.y;
      double dz = new_position_world.z - lantern.position.z;
      double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (distance < min_distance) {
        min_distance = distance;
        closest_idx = i;
      }
    }

    if (closest_idx >= 0) {
      // Update existing lantern - calculate true average of all detections
      auto& lantern = detected_lanterns_[closest_idx];
      
      // Add new detection to sum
      lantern.sum.x += new_position_world.x;
      lantern.sum.y += new_position_world.y;
      lantern.sum.z += new_position_world.z;
      lantern.update_count++;
      
      // Calculate true average position
      lantern.position.x = lantern.sum.x / lantern.update_count;
      lantern.position.y = lantern.sum.y / lantern.update_count;
      lantern.position.z = lantern.sum.z / lantern.update_count;
    } else {
      // Add new lantern (no existing lantern within threshold distance in world coordinates)
      DetectedLantern new_lantern;
      new_lantern.position = new_position_world;
      new_lantern.sum = new_position_world;  // Initialize sum
      new_lantern.update_count = 1;
      detected_lanterns_.push_back(new_lantern);
    }
  }

  // Remove lanterns that haven't been confirmed (not enough updates)
  void cleanup_unconfirmed_lanterns()
  {
    // Remove lanterns with too few updates
    detected_lanterns_.erase(
      std::remove_if(detected_lanterns_.begin(), detected_lanterns_.end(),
        [this](const DetectedLantern& lantern) {
          return lantern.update_count < min_updates_to_confirm_;
        }),
      detected_lanterns_.end());
  }

  // Publish all detected lanterns as visual markers (in world frame)
  void publish_lantern_locations(const rclcpp::Time& timestamp)
  {
    visualization_msgs::msg::MarkerArray marker_array;

    for (size_t i = 0; i < detected_lanterns_.size(); ++i) {
      const auto& lantern = detected_lanterns_[i];
      
      // Only publish confirmed lanterns (with enough updates)
      if (lantern.update_count < min_updates_to_confirm_) {
        continue;
      }
      
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = timestamp;
      marker.header.frame_id = "world";  // Publish in world frame
      marker.ns = "lanterns";
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      
      // Set position (already in world coordinates)
      marker.pose.position = lantern.position;
      marker.pose.orientation.w = 1.0;
      
      // Set scale (0.3m diameter sphere)
      marker.scale.x = 5.0;
      marker.scale.y = 5.0;
      marker.scale.z = 5.0;
      
      // Set color (orange/yellow for lantern, brighter with more updates)
      float brightness = std::min(1.0f, 0.5f + 0.1f * lantern.update_count);
      marker.color.r = 1.0;
      marker.color.g = 0.6 * brightness;
      marker.color.b = 0.0;
      marker.color.a = 0.8;  // Slight transparency
      
      // Infinite lifetime for confirmed lanterns (they persist until manually removed)
      marker.lifetime = rclcpp::Duration::from_seconds(0);
      
      marker_array.markers.push_back(marker);
    }

    lantern_locations_pub_->publish(marker_array);
  }

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::msg::Image> semantic_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Camera info subscriber
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;

  // TF2 for coordinate transformations
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lantern_locations_pub_;

  // Pre-allocated buffers
  sensor_msgs::msg::Image::SharedPtr filtered_depth_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;

  // Camera intrinsics (from CameraInfo subscription)
  bool have_camera_info_ = false;
  double focal_length_x_;
  double focal_length_y_;
  double principal_point_x_;
  double principal_point_y_;
  double depth_scale_;

  // Lantern tracking parameters and state
  std::vector<DetectedLantern> detected_lanterns_;
  double matching_distance_threshold_;
  int frame_skip_;
  int frame_counter_ = 0;
  int min_updates_to_confirm_;
  int cleanup_counter_ = 0;
};

}  // namespace lantern_location

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lantern_location::LanternLocationNode>());
  rclcpp::shutdown();
  return 0;
}
