import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. (Optional) Depth to PointCloud Conversion Node
    # Only needed if your camera does not publish a PointCloud2 directly.
    # We use a composable node container for efficiency, but a standard node works for simple setups.
    depth_proc_node = Node(
        package='depth_image_proc',
        executable='point_cloud_xyz_node',
        name='point_cloud_xyz_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        # Remap input topics to your camera's topics
        remappings=[
            ('image_rect', 'realsense/depth/image'), # Input Depth
            ('camera_info', 'realsense/depth/camera_info'),   # Input Camera Info
            ('points', 'realsense/depth/points')              # Output PointCloud
        ]
    )

    # 2. Octomap Server Node
    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,              # Must match rest of system (no /clock published)
            'resolution': 2.0,                 # Voxel size (meters)
            'frame_id': 'world',                # Fixed frame (map/world)
            'base_frame_id': 'true_body',       # Robot base frame
            'sensor_model.max_range': 60.0,     # Reduced from 60m — large range + 2m voxels amplifies TF timing errors
            'sensor_model.hit': 0.8,            # Require stronger evidence to mark occupied (default 0.7)
            'sensor_model.miss': 0.3,           # Faster free-space clearing to recover from transient phantom voxels
            'filter_ground': False,             # Filter out floor plane?
            'filter_speckles': True,            # Remove isolated voxels (phantom artifacts from TF hiccups)
            'latch': False                       
        }],
        remappings=[
            ('cloud_in', 'realsense/depth/points') # Subscribe to the pointcloud
        ]
    )

    return LaunchDescription([
        depth_proc_node,
        octomap_node
    ])