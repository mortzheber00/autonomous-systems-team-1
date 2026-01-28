#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_srvs/srv/empty.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "TreeNode.h"

/**
 * RRT3DNode - 3D Rapid Exploring Random Tree Planner
 * 
 * This ROS 2 node implements the RRT algorithm for 3D path planning.
 * 
 * Algorithm Overview:
 * 1. Start with a tree containing only the current drone position
 * 2. Repeat until goal found or node limit reached:
 *    - Generate random point in configuration space
 *    - Find nearest node in tree to random point
 *    - Extend tree toward random point by step_size
 *    - If new node is close enough to goal, connect to goal and publish path
 * 
 * ROS Interface (Subscribers - Continuous):
 * - /goal_point (geometry_msgs/PoseStamped) - Goal from local planner
 * - /odom or /true_pose (nav_msgs/Odometry or geometry_msgs/PoseStamped) - Current drone position
 * 
 * ROS Interface (Publishers):
 * - /path (nav_msgs/Path) - Outputs the planned trajectory
 * - /rrt_tree (visualization_msgs/MarkerArray) - RRT tree visualization
 * - /rrt_path (visualization_msgs/Marker) - Final path visualization
 */
class RRT3DNode : public rclcpp::Node
{
public:
  RRT3DNode()
      : Node("RRT3DNode"), step_size(0.5), goal_tolerance(1), map_sub_mode(false), obstacle_sub_mode(false),
        node_limit(10000), wall_confidence(50)
  {
    // Create publisher for the planned path
    path_publisher = create_publisher<nav_msgs::msg::Path>("path", 10);
    
    // Create publisher for RRT visualization (tree nodes and edges)
    tree_marker_publisher = create_publisher<visualization_msgs::msg::MarkerArray>("rrt_tree", 10);
    
    // Create publisher for path visualization
    path_marker_publisher = create_publisher<visualization_msgs::msg::Marker>("rrt_path", 10);
    
    // Subscribe to frontier goal points from local planner (octomap-based exploration)
    frontier_subscriber = create_subscription<geometry_msgs::msg::Point>(
      "/frontier_goal",
      10,
      std::bind(&RRT3DNode::frontier_callback, this, std::placeholders::_1)
    );
    
    // Subscribe to current drone position (odometry)
    odom_subscriber = create_subscription<nav_msgs::msg::Odometry>(
      "/true_pose",  // Change to /odom or your drone's odometry topic
      10,
      std::bind(&RRT3DNode::odom_callback, this, std::placeholders::_1)
    );
    
    RCLCPP_INFO(get_logger(), "RRT3DNode initialized. Waiting for frontier points on /frontier_goal...");
  }

private:
  // Configuration parameters
  const float step_size;            // Distance to extend tree at each iteration
  const float goal_tolerance;       // Tolerance for reaching goal (stopping criterion)
  const bool map_sub_mode;          // Use map subscription (for obstacle avoidance)
  const bool obstacle_sub_mode;     // Use obstacle subscription
  const size_t node_limit;          // Maximum number of nodes before stopping
  const int wall_confidence;        // Confidence threshold for detecting walls
  
  // Current state (updated by subscribers)
  std::vector<float> start_position = {0.0f, 0.0f, 0.0f};  // Current drone position
  std::vector<float> goal_position = {5.0f, 5.0f, 5.0f};   // Goal from local planner
  bool has_goal = false;                                     // Whether we have a new goal
  bool is_planning = false;                                  // Prevent concurrent planning
  
  // RRT tree data structures
  std::vector<int> map_size = {10, 10, 10};  // Configuration space bounds
  std::vector<TreeNode *> node_list;         // All nodes in the tree
  TreeNode *root_node;                       // Root node (start position)
  
  // ROS communication
  nav_msgs::msg::Path path;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_marker_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_publisher;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr frontier_subscriber;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber;

  /**
   * Callback for frontier goal point subscriber
   * 
   * Called whenever local planner publishes a new frontier goal.
   * Triggers RRT planning to the frontier point.
   */
  void frontier_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received frontier goal: (%.2f, %.2f, %.2f)",
                msg->x, msg->y, msg->z);
    
    // Update goal position from frontier point
    goal_position[0] = msg->x;
    goal_position[1] = msg->y;
    goal_position[2] = msg->z;
    has_goal = true;
    
    // Start planning if not already planning
    if (!is_planning)
    {
      is_planning = true;
      plan_path();
      is_planning = false;
    }
  }

  /**
   * Callback for odometry/position subscriber
   * 
   * Called whenever drone position is updated.
   * Updates the start position for RRT planning.
   */
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Update current drone position as RRT start
    start_position[0] = msg->pose.pose.position.x;
    start_position[1] = msg->pose.pose.position.y;
    start_position[2] = msg->pose.pose.position.z;
  }

  /**
   * Main RRT planning function
   * 
   * This is called whenever a new goal is received.
   * Generates a path from current position to goal.
   */
  void plan_path()
  {
    RCLCPP_INFO(get_logger(), "Planning RRT from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)",
                start_position[0], start_position[1], start_position[2],
                goal_position[0], goal_position[1], goal_position[2]);

    // Initialize tree with current position
    TreeNode *root_node = new TreeNode(start_position, nullptr);
    
    // Clear previous tree
    for (TreeNode *node : node_list)
    {
      delete node;
    }
    node_list = {root_node};
    
    // RRT main loop: expand tree until goal found or node limit reached
    while (node_list.size() < node_limit)
    {
      // Step 1: Generate random point in configuration space
      const float rand_x = rand() % map_size[0];
      const float rand_y = rand() % map_size[1];
      const float rand_z = rand() % map_size[2];
      
      float min_distance = INFINITY;
      std::vector<float> min_node_vec;
      TreeNode *nearest_node;
      
      // Step 2: Find nearest node in tree to random point
      for (TreeNode *node : node_list)
      {
        // Check if this node is close enough to goal
        std::vector<float> goal_vec = {goal_position[0] - node->val[0],
                                       goal_position[1] - node->val[1],
                                       goal_position[2] - node->val[2]};
        const float goal_distance = sqrt(pow(goal_vec[0], 2) +
                                         pow(goal_vec[1], 2) +
                                         pow(goal_vec[2], 2));
        
        // Goal found! Connect to goal and publish path
        if (goal_distance < goal_tolerance)
        {
          TreeNode *goal_node = new TreeNode(goal_position, node);
          node->add_child(goal_node);
          node_list.push_back(goal_node);
          RCLCPP_INFO(get_logger(), "Path found!");
          visualize_rrt_tree();   // Show the complete tree
          visualize_path();       // Show the final path
          publish_path();
          return;
        }
        
        // Calculate distance from node to random point
        std::vector<float> node_vec = {node->val[0] - rand_x,
                                       node->val[1] - rand_y,
                                       node->val[2] - rand_z};
        const float distance = sqrt(pow(node_vec[0], 2) +
                                    pow(node_vec[1], 2) +
                                    pow(node_vec[2], 2));
        
        // Keep track of nearest node
        if (distance < min_distance)
        {
          min_node_vec = node_vec;
          min_distance = distance;
          nearest_node = node;
        }
      }
      
      // Step 3: Extend tree toward random point by step_size
      const float new_node_x = nearest_node->val[0] - (min_node_vec[0] / min_distance) * step_size;
      const float new_node_y = nearest_node->val[1] - (min_node_vec[1] / min_distance) * step_size;
      const float new_node_z = nearest_node->val[2] - (min_node_vec[2] / min_distance) * step_size;
      std::vector<float> new_node_position = {new_node_x, new_node_y, new_node_z};
      
      // Create new node and add to tree
      TreeNode *new_node = new TreeNode(new_node_position, nearest_node);
      nearest_node->add_child(new_node);
      node_list.push_back(new_node);

      // Periodically visualize tree progress (every 100 nodes)
      if (node_list.size() % 100 == 0)
      {
        visualize_rrt_tree();
      }
    }
    
    // Node limit reached without finding path
    RCLCPP_INFO(get_logger(), "Path not found within node limit");
    return;
  }

  /**
   * Visualize the RRT tree in RViz
   * 
   * Publishes:
   * - Nodes as small blue spheres
   * - Edges as white lines connecting parent to child
   * - Start as green sphere
   * - Goal as red sphere
   */
  void visualize_rrt_tree()
  {
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;

    // Publish all nodes as spheres
    for (TreeNode *node : node_list)
    {
      visualization_msgs::msg::Marker node_marker;
      node_marker.header.frame_id = "map";
      node_marker.header.stamp = get_clock()->now();
      node_marker.id = marker_id++;
      node_marker.type = visualization_msgs::msg::Marker::SPHERE;
      node_marker.action = visualization_msgs::msg::Marker::ADD;
      node_marker.pose.position.x = node->val[0];
      node_marker.pose.position.y = node->val[1];
      node_marker.pose.position.z = node->val[2];
      node_marker.scale.x = node_marker.scale.y = node_marker.scale.z = 0.1;
      node_marker.color.r = 0.0f;
      node_marker.color.g = 0.0f;
      node_marker.color.b = 1.0f;  // Blue for regular nodes
      node_marker.color.a = 0.8f;
      markers.markers.push_back(node_marker);

      // Draw edge from parent to this node
      if (node->parent != nullptr)
      {
        visualization_msgs::msg::Marker edge_marker;
        edge_marker.header.frame_id = "map";
        edge_marker.header.stamp = get_clock()->now();
        edge_marker.id = marker_id++;
        edge_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        edge_marker.action = visualization_msgs::msg::Marker::ADD;
        edge_marker.scale.x = 0.05;  // Line width
        edge_marker.color.r = 1.0f;
        edge_marker.color.g = 1.0f;
        edge_marker.color.b = 1.0f;  // White edges
        edge_marker.color.a = 0.6f;

        geometry_msgs::msg::Point p1, p2;
        p1.x = node->parent->val[0];
        p1.y = node->parent->val[1];
        p1.z = node->parent->val[2];
        p2.x = node->val[0];
        p2.y = node->val[1];
        p2.z = node->val[2];
        edge_marker.points.push_back(p1);
        edge_marker.points.push_back(p2);
        markers.markers.push_back(edge_marker);
      }
    }

    // Highlight start position (green sphere)
    visualization_msgs::msg::Marker start_marker;
    start_marker.header.frame_id = "map";
    start_marker.header.stamp = get_clock()->now();
    start_marker.id = marker_id++;
    start_marker.type = visualization_msgs::msg::Marker::SPHERE;
    start_marker.action = visualization_msgs::msg::Marker::ADD;
    start_marker.pose.position.x = start_position[0];
    start_marker.pose.position.y = start_position[1];
    start_marker.pose.position.z = start_position[2];
    start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.3;
    start_marker.color.r = 0.0f;
    start_marker.color.g = 1.0f;
    start_marker.color.b = 0.0f;  // Green for start
    start_marker.color.a = 1.0f;
    markers.markers.push_back(start_marker);

    // Highlight goal position (red sphere)
    visualization_msgs::msg::Marker goal_marker;
    goal_marker.header.frame_id = "map";
    goal_marker.header.stamp = get_clock()->now();
    goal_marker.id = marker_id++;
    goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
    goal_marker.action = visualization_msgs::msg::Marker::ADD;
    goal_marker.pose.position.x = goal_position[0];
    goal_marker.pose.position.y = goal_position[1];
    goal_marker.pose.position.z = goal_position[2];
    goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.3;
    goal_marker.color.r = 1.0f;
    goal_marker.color.g = 0.0f;
    goal_marker.color.b = 0.0f;  // Red for goal
    goal_marker.color.a = 1.0f;
    markers.markers.push_back(goal_marker);

    tree_marker_publisher->publish(markers);
  }

  /**
   * Visualize the final path as a thick line
   */
  void visualize_path()
  {
    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "map";
    path_marker.header.stamp = get_clock()->now();
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.scale.x = 0.1;  // Line width
    path_marker.color.r = 0.0f;
    path_marker.color.g = 1.0f;
    path_marker.color.b = 0.0f;  // Green for final path
    path_marker.color.a = 1.0f;

    // Add path points in reverse order (from goal to start)
    TreeNode *current_node = node_list.back();
    while (current_node != nullptr)
    {
      geometry_msgs::msg::Point p;
      p.x = current_node->val[0];
      p.y = current_node->val[1];
      p.z = current_node->val[2];
      path_marker.points.push_back(p);
      current_node = current_node->parent;
    }

    path_marker_publisher->publish(path_marker);
  }

  /**
   * Publish the path from start to goal
   * 
   * Backtracks from the goal node through parent pointers to reconstruct
   * the full path, then publishes it as a nav_msgs::Path message.
   */
  void publish_path()
  {
    path.poses.clear();
    path.header.frame_id = "map";
    path.header.stamp = get_clock()->now();
    
    // Backtrack from goal node to start using parent pointers
    TreeNode *current_node = node_list.back();
    while (current_node->parent != nullptr)
    {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.header.stamp = get_clock()->now();
      pose.pose.position.x = current_node->val[0];
      pose.pose.position.y = current_node->val[1];
      pose.pose.position.z = current_node->val[2];
      path.poses.push_back(pose);
      current_node = current_node->parent;
    }
    path_publisher->publish(path);
  }

  /**
   * RRT Algorithm Implementation
   * 
   * Called when /run_rrt service is invoked
   */
  void rrt_service_callback(std_srvs::srv::Empty::Request::SharedPtr,
                            std_srvs::srv::Empty::Response::SharedPtr)
  {
    RCLCPP_INFO(get_logger(), "Running RRT");
    
    // Initialize tree with start position
    TreeNode *root_node = new TreeNode(start_position, nullptr);
    
    // Clear previous tree
    for (TreeNode *node : node_list)
    {
      delete node;
    }
    node_list = {root_node};
    
    // RRT main loop: expand tree until goal found or node limit reached
    while (node_list.size() < node_limit)
    {
      // Step 1: Generate random point in configuration space
      const float rand_x = rand() % map_size[0];
      const float rand_y = rand() % map_size[1];
      const float rand_z = rand() % map_size[2];
      
      float min_distance = INFINITY;
      std::vector<float> min_node_vec;
      TreeNode *nearest_node;
      
      // Step 2: Find nearest node in tree to random point
      for (TreeNode *node : node_list)
      {
        // Check if this node is close enough to goal
        std::vector<float> goal_vec = {goal_position[0] - node->val[0],
                                       goal_position[1] - node->val[1],
                                       goal_position[2] - node->val[2]};
        const float goal_distance = sqrt(pow(goal_vec[0], 2) +
                                         pow(goal_vec[1], 2) +
                                         pow(goal_vec[2], 2));
        
        // Goal found! Connect to goal and publish path
        if (goal_distance < goal_tolerance)
        {
          TreeNode *goal_node = new TreeNode(goal_position, node);
          node->add_child(goal_node);
          node_list.push_back(goal_node);
          RCLCPP_INFO(get_logger(), "Path found");
          visualize_rrt_tree();   // Show the complete tree
          visualize_path();       // Show the final path
          publish_path();
          return;
        }
        
        // Calculate distance from node to random point
        std::vector<float> node_vec = {node->val[0] - rand_x,
                                       node->val[1] - rand_y,
                                       node->val[2] - rand_z};
        const float distance = sqrt(pow(node_vec[0], 2) +
                                    pow(node_vec[1], 2) +
                                    pow(node_vec[2], 2));
        
        // Keep track of nearest node
        if (distance < min_distance)
        {
          min_node_vec = node_vec;
          min_distance = distance;
          nearest_node = node;
        }
      }
      
      // Step 3: Extend tree toward random point by step_size
      const float new_node_x = nearest_node->val[0] - (min_node_vec[0] / min_distance) * step_size;
      const float new_node_y = nearest_node->val[1] - (min_node_vec[1] / min_distance) * step_size;
      const float new_node_z = nearest_node->val[2] - (min_node_vec[2] / min_distance) * step_size;
      std::vector<float> new_node_position = {new_node_x, new_node_y, new_node_z};
      
      // Create new node and add to tree
      TreeNode *new_node = new TreeNode(new_node_position, nearest_node);
      nearest_node->add_child(new_node);
      node_list.push_back(new_node);

      // Periodically visualize tree progress (every 100 nodes)
      if (node_list.size() % 100 == 0)
      {
        visualize_rrt_tree();
      }
    }
    
    // Node limit reached without finding path
    RCLCPP_INFO(get_logger(), "Path not found");
    return;
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RRT3DNode>());
  rclcpp::shutdown();
  return 0;
}