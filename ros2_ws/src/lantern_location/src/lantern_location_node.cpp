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
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

// Define the namespace for this node
namespace lantern_location
{

// Sync policy for approximate time synchronization
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
  sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

// Structure to track detected lanterns
struct DetectedLantern
{
  geometry_msgs::msg::Point position;  // Average position in camera frame
  int update_count;                     // Number of times this lantern was updated
  rclcpp::Time last_seen;              // Last time this lantern was detected
};

class LanternLocationNode : public rclcpp::Node
{
public:
  LanternLocationNode()
  : rclcpp::Node("lantern_location_node")
  {
    // Depth scale parameter (conversion from depth units to meters)
    this->declare_parameter("depth_scale", 0.001);  // mm to meters
    depth_scale_ = this->get_parameter("depth_scale").as_double();

    // Processing frame skip parameter (process every Nth frame)
    this->declare_parameter("frame_skip", 3);  // Process every 3rd frame by default
    frame_skip_ = this->get_parameter("frame_skip").as_int();

    // Lantern matching distance threshold (meters)
    this->declare_parameter("matching_distance_threshold", 5.0);
    matching_distance_threshold_ = this->get_parameter("matching_distance_threshold").as_double();

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
          RCLCPP_INFO(get_logger(), "Received camera intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f",
            focal_length_x_, focal_length_y_, principal_point_x_, principal_point_y_);
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

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "Processing synchronized images (%ux%u)", depth_msg->width, depth_msg->height);

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
    if (cloud_->width != width || cloud_->height != height) {
      setup_pointcloud_fields(*cloud_, width, height, frame_id);
    }
    cloud_->header.stamp = timestamp;
    cloud_->header.frame_id = frame_id;

    // Compute 3D points in camera frame and accumulate for averaging
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

        *iter_x = x;
        *iter_y = y;
        *iter_z = z;

        // Accumulate for average calculation
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
      geometry_msgs::msg::Point avg_position;
      avg_position.x = sum_x / valid_points;
      avg_position.y = sum_y / valid_points;
      avg_position.z = sum_z / valid_points;

      RCLCPP_INFO(get_logger(), 
        "Detected lantern at average position: (%.3f, %.3f, %.3f) from %d points",
        avg_position.x, avg_position.y, avg_position.z, valid_points);

      // Update or add lantern to tracking list
      update_lantern_list(avg_position, timestamp);
      
      // Publish all detected lanterns
      publish_lantern_locations(frame_id, timestamp);
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "No valid lantern points detected in this frame");
    }
  }

  // Update or add a lantern to the tracking list
  void update_lantern_list(const geometry_msgs::msg::Point& new_position,
                          const rclcpp::Time& timestamp)
  {
    // Check if this lantern matches any existing lantern (within threshold distance)
    int closest_idx = -1;
    double min_distance = matching_distance_threshold_;

    for (size_t i = 0; i < detected_lanterns_.size(); ++i) {
      const auto& lantern = detected_lanterns_[i];
      double dx = new_position.x - lantern.position.x;
      double dy = new_position.y - lantern.position.y;
      double dz = new_position.z - lantern.position.z;
      double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (distance < min_distance) {
        min_distance = distance;
        closest_idx = i;
      }
    }

    if (closest_idx >= 0) {
      // Update existing lantern with weighted average
      auto& lantern = detected_lanterns_[closest_idx];
      double weight = 1.0 / (lantern.update_count + 1.0);
      
      lantern.position.x = lantern.position.x * (1.0 - weight) + new_position.x * weight;
      lantern.position.y = lantern.position.y * (1.0 - weight) + new_position.y * weight;
      lantern.position.z = lantern.position.z * (1.0 - weight) + new_position.z * weight;
      lantern.update_count++;
      lantern.last_seen = timestamp;

      RCLCPP_INFO(get_logger(), 
        "Updated lantern #%d at (%.3f, %.3f, %.3f), updates: %d, distance: %.3fm",
        closest_idx, lantern.position.x, lantern.position.y, lantern.position.z,
        lantern.update_count, min_distance);
    } else {
      // Add new lantern
      DetectedLantern new_lantern;
      new_lantern.position = new_position;
      new_lantern.update_count = 1;
      new_lantern.last_seen = timestamp;
      detected_lanterns_.push_back(new_lantern);

      RCLCPP_INFO(get_logger(), 
        "Added new lantern #%zu at (%.3f, %.3f, %.3f)",
        detected_lanterns_.size() - 1, 
        new_position.x, new_position.y, new_position.z);
    }
  }

  // Publish all detected lanterns as visual markers
  void publish_lantern_locations(const std::string& frame_id,
                                 const rclcpp::Time& timestamp)
  {
    visualization_msgs::msg::MarkerArray marker_array;

    for (size_t i = 0; i < detected_lanterns_.size(); ++i) {
      const auto& lantern = detected_lanterns_[i];
      
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = timestamp;
      marker.header.frame_id = frame_id;
      marker.ns = "lanterns";
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      
      // Set position
      marker.pose.position = lantern.position;
      marker.pose.orientation.w = 1.0;
      
      // Set scale (0.3m diameter sphere)
      marker.scale.x = 0.3;
      marker.scale.y = 0.3;
      marker.scale.z = 0.3;
      
      // Set color (orange/yellow for lantern, brighter with more updates)
      float brightness = std::min(1.0f, 0.5f + 0.1f * lantern.update_count);
      marker.color.r = 1.0;
      marker.color.g = 0.6 * brightness;
      marker.color.b = 0.0;
      marker.color.a = 0.8;  // Slight transparency
      
      marker.lifetime = rclcpp::Duration::from_seconds(2.0);
      
      marker_array.markers.push_back(marker);
    }

    lantern_locations_pub_->publish(marker_array);
    
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "Published %zu detected lantern(s)", detected_lanterns_.size());
  }

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::msg::Image> semantic_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Camera info subscriber
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lantern_locations_pub_;

  // Pre-allocated buffers
  sensor_msgs::msg::Image::SharedPtr filtered_depth_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;

  // Camera intrinsics (from CameraInfo subscription)
  bool have_camera_info_ = false;
  double focal_length_x_ = 0.0;
  double focal_length_y_ = 0.0;
  double principal_point_x_ = 0.0;
  double principal_point_y_ = 0.0;
  double depth_scale_ = 0.001;

  // Lantern tracking parameters and state
  std::vector<DetectedLantern> detected_lanterns_;
  double matching_distance_threshold_ = 5.0;  // meters
  int frame_skip_ = 3;  // Process every Nth frame
  int frame_counter_ = 0;
};

}  // namespace lantern_location

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lantern_location::LanternLocationNode>());
  rclcpp::shutdown();
  return 0;
}
