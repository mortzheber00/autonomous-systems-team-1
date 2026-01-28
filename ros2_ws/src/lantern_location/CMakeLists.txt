cmake_minimum_required(VERSION 3.8)
project(lantern_location)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(lantern_location_node src/lantern_location_node.cpp)
ament_target_dependencies(lantern_location_node
  rclcpp sensor_msgs message_filters tf2 tf2_ros tf2_geometry_msgs)

install(TARGETS lantern_location_node DESTINATION lib/${PROJECT_NAME})

ament_package()
