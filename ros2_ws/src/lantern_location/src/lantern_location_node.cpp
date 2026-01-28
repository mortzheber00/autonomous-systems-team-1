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
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Define the namespace for this node
namespace lantern_location
{

// Sync policy for approximate time synchronization
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
  sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

class LanternLocationNode : public rclcpp::Node
{
public:
  LanternLocationNode()
  : rclcpp::Node("lantern_location_node")
  {
    // Depth scale parameter (conversion from depth units to meters)
    this->declare_parameter("depth_scale", 0.001);  // mm to meters
    depth_scale_ = this->get_parameter("depth_scale").as_double();

    // Initialize TF2 for coordinate transformations
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

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
    pointcloud_transformed_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lantern_location/filtered_pointcloud_map", 10);

    // Pre-allocate buffers (will be resized on first use if needed)
    filtered_depth_ = std::make_shared<sensor_msgs::msg::Image>();
    cloud_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud_transformed_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
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

    // Build point cloud (reuse buffers, single pass)
    publish_pointcloud(filtered_depth_, width, height);
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

  void publish_pointcloud(const sensor_msgs::msg::Image::SharedPtr& depth_image,
                          size_t width, size_t height)
  {
    const size_t num_pixels = width * height;

    // Setup point cloud structure (only reallocate if size changed)
    // Use Quadrotor/DepthCamera which exists in the TF tree
    if (cloud_->width != width || cloud_->height != height) {
      setup_pointcloud_fields(*cloud_, width, height, "Quadrotor/DepthCamera");
    }
    cloud_->header.stamp = depth_image->header.stamp;

    // Get transform early (before processing points)
    bool have_transform = false;
    tf2::Transform tf2_transform;
    try {
      auto transform = tf_buffer_->lookupTransform(
        "map", cloud_->header.frame_id, cloud_->header.stamp,
        std::chrono::milliseconds(50));
      tf2::fromMsg(transform.transform, tf2_transform);
      have_transform = true;

      // Prepare transformed cloud buffer
      if (cloud_transformed_->width != width || cloud_transformed_->height != height) {
        setup_pointcloud_fields(*cloud_transformed_, width, height, "map");
      }
      cloud_transformed_->header.stamp = depth_image->header.stamp;
    } catch (const tf2::TransformException& e) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 5000,
        "Transform to map unavailable: %s", e.what());
    }

    // Single pass: compute 3D points and optionally transform
    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_, "z");

    // Only create transformed iterators if we have a valid transform
    std::unique_ptr<sensor_msgs::PointCloud2Iterator<float>> iter_tx, iter_ty, iter_tz;
    if (have_transform) {
      iter_tx = std::make_unique<sensor_msgs::PointCloud2Iterator<float>>(*cloud_transformed_, "x");
      iter_ty = std::make_unique<sensor_msgs::PointCloud2Iterator<float>>(*cloud_transformed_, "y");
      iter_tz = std::make_unique<sensor_msgs::PointCloud2Iterator<float>>(*cloud_transformed_, "z");
    }

    const uint16_t* depth_ptr = reinterpret_cast<const uint16_t*>(depth_image->data.data());
    int valid_points = 0;

    for (size_t i = 0; i < num_pixels; ++i, ++iter_x, ++iter_y, ++iter_z) {
      const uint16_t depth_value = depth_ptr[i];

      if (depth_value == 0) {
        *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
        if (have_transform) {
          **iter_tx = **iter_ty = **iter_tz = std::numeric_limits<float>::quiet_NaN();
          ++(*iter_tx); ++(*iter_ty); ++(*iter_tz);
        }
      } else {
        // Convert pixel to 3D point
        const int u = i % width;
        const int v = i / width;
        const float z = depth_value * depth_scale_;
        const float x = (u - principal_point_x_) * z / focal_length_x_;
        const float y = (v - principal_point_y_) * z / focal_length_y_;

        *iter_x = x;
        *iter_y = y;
        *iter_z = z;

        // Log first 3 valid points for debugging
        if (valid_points < 3) {
          RCLCPP_INFO(get_logger(), "Point %d: pixel(%d,%d) depth=%u -> XYZ(%.3f, %.3f, %.3f)",
            valid_points, u, v, depth_value, x, y, z);
        }

        if (have_transform) {
          tf2::Vector3 pt_transformed = tf2_transform * tf2::Vector3(x, y, z);
          **iter_tx = pt_transformed.x();
          **iter_ty = pt_transformed.y();
          **iter_tz = pt_transformed.z();
          ++(*iter_tx); ++(*iter_ty); ++(*iter_tz);
        }
        ++valid_points;
      }
    }

    pointcloud_pub_->publish(*cloud_);
    if (have_transform) {
      pointcloud_transformed_pub_->publish(*cloud_transformed_);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "Published point cloud: %d valid points / %zu total", valid_points, num_pixels);
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
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_transformed_pub_;

  // TF2
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // Pre-allocated buffers
  sensor_msgs::msg::Image::SharedPtr filtered_depth_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_transformed_;

  // Camera intrinsics (from CameraInfo subscription)
  bool have_camera_info_ = false;
  double focal_length_x_ = 0.0;
  double focal_length_y_ = 0.0;
  double principal_point_x_ = 0.0;
  double principal_point_y_ = 0.0;
  double depth_scale_ = 0.001;
};

}  // namespace lantern_location

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lantern_location::LanternLocationNode>());
  rclcpp::shutdown();
  return 0;
}
