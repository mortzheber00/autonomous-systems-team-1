# Planner Package – Autonomous Cave Exploration

A ROS 2 (Humble) planning stack for autonomous cave exploration with a MAV (Micro Aerial Vehicle). The system uses a **three-node pipeline** — Local Planner → Global Planner → RRT Path Planner — to explore unknown cave environments, detect junctions, backtrack to unexplored branches, and return home when exploration is complete.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Node Descriptions](#node-descriptions)
   - [Local Planner Node](#1-local-planner-node)
   - [Global Planner Node](#2-global-planner-node)
   - [RRT 3D Path Planner](#3-rrt-3d-path-planner)
3. [State Machine](#state-machine)
4. [Algorithms](#algorithms)
   - [Frontier Detection](#frontier-detection)
   - [Volumetric Gain Scoring](#volumetric-gain-scoring)
   - [Direction Gain](#direction-gain)
   - [Junction Detection](#junction-detection)
   - [Breadcrumb Trail & Dijkstra Backtracking](#breadcrumb-trail--dijkstra-backtracking)
   - [RRT Path Planning](#rrt-path-planning)
   - [Trajectory Publishing & Collision Detection](#trajectory-publishing--collision-detection)
5. [ROS Topics](#ros-topics)
6. [ROS Parameters](#ros-parameters)
7. [RViz Visualization](#rviz-visualization)
8. [Dependencies](#dependencies)
9. [Building & Running](#building--running)

---

## System Architecture

```
┌─────────────────┐   /frontier_goals    ┌──────────────────┐   /global_goal     ┌─────────────────┐
│  Local Planner   │ ──────────────────▶  │  Global Planner   │ ──────────────────▶ │  RRT 3D Planner  │
│  (2 Hz loop)     │                      │  (1 Hz loop)      │                     │  (10 Hz publish) │
│                  │                      │                   │  /rrt_status        │                  │
│  Frontier-based  │                      │  State machine    │ ◀────────────────── │  RRT + collision │
│  exploration     │                      │  Junction detect  │                     │  avoidance       │
└────────┬─────────┘                      └────────┬──────────┘                     └────────┬─────────┘
         │                                         │                                         │
         │ /octomap_full                           │ /octomap_full                           │ /octomap_full
         │ /current_state_est                      │ /current_state_est                      │ /current_state_est
         ▼                                         ▼                                         │
   ┌──────────┐                              ┌──────────┐                                    │ /command/trajectory
   │ OctoMap  │                              │ OctoMap  │                                    ▼
   │ Server   │                              │ Server   │                              ┌──────────┐
   └──────────┘                              └──────────┘                              │Controller│
                                                                                       │(1000 Hz) │
                                                                                       └──────────┘
```

**Data flow:**

1. **Local Planner** continuously discovers frontier points in the OctoMap and publishes the top 5 highest-gain frontiers.
2. **Global Planner** receives frontiers, monitors exploration progress, detects junctions, decides when to backtrack, and publishes high-priority goals via `/global_goal`.
3. **RRT 3D Planner** plans collision-free paths to either local frontiers or global goals (global goals take priority). Publishes trajectory waypoints to the flight controller.

---

## Node Descriptions

### 1. Local Planner Node

**File:** `src/local_planner_node.cpp` (462 lines)  
**Executable:** `local_planner_node`  
**Purpose:** Frontier-based local exploration planner running at **2 Hz**.

The local planner iterates over the OctoMap to find *frontier points* — free cells at the boundary between known and unknown space — and scores them by a combination of volumetric gain and directional bias. Only the top 5 frontiers passing a minimum gain threshold are published.

#### Key Behaviour

- Scans OctoMap leaves within a **60 m** search radius of the drone
- Identifies free voxels with at least one unknown neighbour (excluding those adjacent to obstacles)
- Scores each frontier by `volumetric_gain × direction_gain`
- Filters out frontiers with combined gain below **1.3** (`min_publish_gain_`)
- Publishes up to **5** best frontiers as a `geometry_msgs/PoseArray`
- Publishes corresponding RViz markers (green cubes with gain labels)

---

### 2. Global Planner Node

**File:** `src/global_planner_node.cpp` (1209 lines)  
**Executable:** `global_planner_node`  
**Purpose:** High-level state machine orchestrating the full exploration lifecycle.

The global planner manages four exploration states, detects junctions in the cave, maintains a breadcrumb trail, and uses Dijkstra's algorithm to compute shortest backtrack routes. It publishes high-priority navigation goals to the RRT planner via `/global_goal`.

#### Key Data Structures

| Structure | Fields | Description |
|-----------|--------|-------------|
| `JunctionArm` | `direction`, `target`, `exploration_gain`, `explored` | One arm/branch of a junction |
| `Junction` | `id`, `position`, `arms[]`, `last_updated`, `fully_explored` | A confirmed cave junction |
| `JunctionCandidate` | `position`, `detection_count`, `first_seen`, `last_seen`, `arm_clusters[]`, `arm_directions[]` | Unconfirmed junction awaiting temporal confirmation |
| `ExplorationState` | `EXPLORING`, `BACKTRACKING`, `RETURNING_HOME`, `DONE` | FSM states |

#### Key Behaviour

- **Gain-validated frontier timer**: Only resets the "no frontier" timer when incoming frontiers have gain ≥ `min_frontier_gain_for_timer_` (0.3), preventing dead-end frontiers from keeping the drone stuck.
- **Stagnation detection**: Monitors OctoMap leaf-node count growth. If growth < `min_volume_delta_` (5 nodes) for `stagnation_timeout_` (20 s), triggers backtracking.
- **Junction detection with temporal confirmation**: Clusters frontier directions, checks angular separation, and requires multiple detections (`junction_confirm_count_`) before confirming a junction.
- **Breadcrumb trail**: Records drone positions every `breadcrumb_spacing_` (30 m) during exploration.
- **Dijkstra backtrack routing**: Builds a graph over breadcrumbs (chain edges + shortcut edges for nearby non-consecutive crumbs), finds shortest path, and creates multi-hop waypoint queue.
- **Multi-hop waypoint navigation**: Sends one waypoint at a time to RRT, waits for `/rrt_status` confirmation, then advances to the next.

---

### 3. RRT 3D Path Planner

**File:** `src/rrt3D.cpp` (in `multidim_rrt_cpp` package, 962 lines)  
**Executable:** `rrt3D`  
**Purpose:** 3D Rapidly-exploring Random Tree (RRT) path planner with OctoMap collision checking.

The RRT planner generates collision-free paths through the OctoMap. It supports two goal sources — local frontiers and global goals — with global goals taking strict priority. Planned paths are converted into timed trajectory waypoints and published at 10 Hz.

#### Key Behaviour

- **Goal priority**: Global goals (`/global_goal`) override local frontier goals. When a global goal arrives, any active local path is cancelled immediately.
- **80% goal bias**: During tree expansion, 80% of samples are directed toward the goal; 20% are random (±100 m XY, ±10 m Z).
- **Safety radius with gradual onset**: 4 m spherical safety zone around each node, linearly reduced near the start position (0.5 m at origin → full radius at ≥ 4 m from start) to prevent false collisions when the drone is near mapped obstacles.
- **5-second timeout** and **5000-node limit** per planning attempt.
- **Active collision detection**: At 10 Hz, checks remaining path segments against the latest OctoMap. If a collision is detected, triggers automatic replanning.
- **Waypoint following**: Publishes `trajectory_msgs/MultiDOFJointTrajectory` with computed yaw (pointing toward next waypoint) and velocity-based timing.

---

## State Machine

The global planner operates a four-state finite state machine:

```
                    ┌───────────────────────────┐
                    │        EXPLORING           │
                    │  (follow local frontiers)  │
                    └─────┬───────────┬──────────┘
                          │           │
          no frontiers OR │           │ no junctions left
          stagnation      │           │ with unexplored arms
                          ▼           ▼
                    ┌──────────┐  ┌──────────────┐
                    │BACKTRACK │  │RETURNING_HOME │
                    │ ING      │  │(-360, 0, 16)  │
                    └────┬─────┘  └───────┬───────┘
                         │                │
                 arrived │                │ arrived
                at arm   │                │ at home
                         ▼                ▼
                    ┌──────────┐     ┌─────────┐
                    │EXPLORING │     │  DONE   │
                    │ (resume) │     │         │
                    └──────────┘     └─────────┘
```

### State Transitions

| From | To | Trigger |
|------|----|---------|
| `EXPLORING` | `BACKTRACKING` | No valid frontiers for `no_frontier_timeout_sec` (12 s) **OR** map stagnation for `stagnation_timeout_sec` (20 s), **AND** an unexplored junction arm exists |
| `EXPLORING` | `RETURNING_HOME` | Same timeout/stagnation trigger but no unexplored junctions remain |
| `BACKTRACKING` | `EXPLORING` | Drone arrives at the target junction arm (within `goal_reached_tolerance_m`) |
| `RETURNING_HOME` | `DONE` | Drone arrives at home position (-360.0, 0.0, 16.0) |

---

## Algorithms

### Frontier Detection

**Node:** Local Planner  
**Method:** `get_frontier_points()`

1. Iterate all OctoMap leaf nodes within `search_radius_` (60 m) of the drone.
2. For each **free** voxel (occupancy < `occupied_threshold_` 0.5):
   - Check 6 face-adjacent neighbours (±x, ±y, ±z).
   - If any neighbour is **unknown** (not in the tree) → candidate frontier.
   - If any neighbour is **occupied** → reject (too close to wall).
3. Return all valid frontier coordinates.

### Volumetric Gain Scoring

**Node:** Local Planner  
**Method:** `calculate_volumetric_gain()`

For each frontier point:

1. Define a bounding box of `exploration_radius_` (10 m) around the point.
2. Count **unknown** vs **total** voxels in the box → `unknown_ratio`.
3. **Wall distance penalty**: Raycast in 26 directions (all face/edge/corner neighbours). The minimum distance to an obstacle determines a penalty:
   - Distance ≥ 4 m → multiplier = **1.0** (no penalty)
   - Distance < 4 m → multiplier = **0.001** (effectively filtered out)
4. Final gain = `unknown_ratio × wall_distance_multiplier`

### Direction Gain

**Node:** Local Planner  
**Method:** `calculate_direction_gain()`

Biases frontier selection toward the drone's current direction of travel:

1. Maintain a **moving average** of the last 10 movement direction samples.
2. Compute the **dot product** between the averaged movement direction and the vector from the drone to the frontier.
3. Map from [-1, 1] → [0, 1] using `(dot + 1) / 2`.

The final frontier score used for ranking is: **`volumetric_gain × direction_gain`**

### Junction Detection

**Node:** Global Planner  
**Method:** `try_detect_junction()`

Detects cave junctions (forks/branches) from frontier distributions:

1. **Pre-filter**: Only consider frontiers at least **8 m** away from the drone (avoids noise from nearby walls).
2. **Angular clustering**: Compute direction vectors from drone to each frontier. Greedily cluster directions within `junction_detect_angle_deg_` (60°) of each other.
3. **Minimum clusters**: Require ≥ 3 distinct directional clusters (a junction must have at least 3 branches).
4. **Cluster separation check**: Verify cluster centroids are ≥ 45° apart.
5. **Proximity check**: Reject if within `junction_proximity_m_` (30 m) of an already-confirmed junction.
6. **Temporal confirmation** (via `JunctionCandidate`):
   - Match new detections to existing candidates within `candidate_merge_radius_` (30 m).
   - Increment `detection_count` on match; create new candidate otherwise.
   - Confirm when `detection_count ≥ junction_confirm_threshold_` (default: 0 → immediate confirmation).
7. On confirmation: create `Junction` with arms pointing toward each cluster centroid, with exploration gain evaluated from the OctoMap.

### Breadcrumb Trail & Dijkstra Backtracking

**Node:** Global Planner  
**Method:** `build_backtrack_route()`

When the drone needs to return to a junction:

1. **Breadcrumb recording**: During `EXPLORING`, the drone's position is saved every `breadcrumb_spacing_` (30 m) of travel.
2. **Graph construction**: Build an adjacency list over all breadcrumbs:
   - **Chain edges**: Consecutive breadcrumbs (index _i_ ↔ _i+1_), weighted by Euclidean distance.
   - **Shortcut edges**: Any two non-consecutive breadcrumbs within `1.5 × breadcrumb_spacing_` (45 m) of each other, enabling loop shortcuts.
3. **Source/target selection**: Find the breadcrumbs closest to the drone (source) and closest to the target junction (target).
4. **Dijkstra's shortest path**: Standard priority-queue Dijkstra over the weighted graph.
5. **Downsampling**: Keep every other waypoint from the result for smoother navigation.
6. **Multi-hop execution**: Waypoints are queued and sent to the RRT one at a time. On `/rrt_status` confirmation of arrival, the next waypoint is dispatched.

This approach correctly handles **loop shortcuts** — when the cave forms a loop, the Dijkstra graph finds the shorter path around rather than retracing the full exploration route.

### RRT Path Planning

**Node:** RRT 3D Planner  
**Method:** `plan_path_internal()`

Core RRT algorithm:

1. **Initialise** tree with the drone's current position as root.
2. **Sample**: With 80% probability, sample the goal point; with 20%, sample a random point in (±100 m XY, ±10 m Z relative to current position).
3. **Nearest node**: Find the tree node closest to the sample (linear scan).
4. **Extend**: Step from the nearest node toward the sample by `step_size` (2.0 m).
5. **Collision check**: Verify the new node and the connecting edge are collision-free against the OctoMap:
   - `is_in_collision()`: Check a spherical volume of `safety_radius_` (4.0 m) using OctoMap bounding-box iteration. Safety radius is **linearly reduced near the start** (minimum 0.5 m at the origin) to allow takeoff near mapped surfaces.
   - `is_edge_collision_free()`: Sample points along the edge every 1.0 m, each checked for collision.
6. **Goal check**: If the new node is within `goal_tolerance` (2.0 m) of the goal, extract the path.
7. **Termination**: Stop after reaching the goal, exceeding `node_limit` (5000), or `rrt_timeout_` (5.0 s).
8. **Path post-processing**: Downsample by factor of 2 (keep every other node) for smoother trajectories.

### Trajectory Publishing & Collision Detection

**Node:** RRT 3D Planner  
**Method:** `trajectory_publishing_loop()` (10 Hz)

1. Publish remaining waypoints as a `trajectory_msgs/MultiDOFJointTrajectory`.
2. Compute **yaw** for each waypoint pointing toward the next waypoint using `atan2`.
3. Compute **timing** based on `desired_velocity` (15.0 m/s) and inter-waypoint distances.
4. **Waypoint reached check**: If the drone is within `goal_tolerance` (2.0 m) of the current target waypoint, advance to the next.
5. **Active collision detection**: For each remaining path segment, call `is_edge_collision_free()`. If a collision is detected (due to updated OctoMap), trigger automatic **replanning**.
6. On path completion, publish `"reached"` on `/rrt_status`.

---

## ROS Topics

### Local Planner Node

| Direction | Topic | Type | Description |
|-----------|-------|------|-------------|
| **Sub** | `/octomap_full` | `octomap_msgs/Octomap` | Full OctoMap from mapping node |
| **Sub** | `/current_state_est` | `nav_msgs/Odometry` | Drone odometry (position + orientation) |
| **Pub** | `/frontier_goals` | `geometry_msgs/PoseArray` | Top 5 frontier goals (scored & filtered) |
| **Pub** | `/frontier_goals_markers` | `visualization_msgs/MarkerArray` | RViz frontier markers |

### Global Planner Node

| Direction | Topic | Type | Description |
|-----------|-------|------|-------------|
| **Sub** | `/frontier_goals` | `geometry_msgs/PoseArray` | Frontiers from local planner |
| **Sub** | `/octomap_full` | `octomap_msgs/Octomap` | Full OctoMap for gain evaluation |
| **Sub** | `/current_state_est` | `nav_msgs/Odometry` | Drone odometry |
| **Sub** | `/rrt_status` | `std_msgs/String` | Path completion status from RRT (`"reached"`) |
| **Pub** | `/global_goal` | `geometry_msgs/PoseArray` | High-priority goal for RRT (backtrack/return) |
| **Pub** | `/exploration_state` | `std_msgs/String` | Current state (`EXPLORING`, `BACKTRACKING`, etc.) |
| **Pub** | `/global_frontier_markers` | `visualization_msgs/MarkerArray` | RViz markers for junctions, arms, breadcrumbs, home |

### RRT 3D Planner Node

| Direction | Topic | Type | Description |
|-----------|-------|------|-------------|
| **Sub** | `/frontier_goals` | `geometry_msgs/PoseArray` | Local frontier goals (low priority) |
| **Sub** | `/global_goal` | `geometry_msgs/PoseArray` | Global planner goals (high priority, overrides local) |
| **Sub** | `/current_state_est` | `nav_msgs/Odometry` | Drone odometry |
| **Sub** | `/octomap_full` | `octomap_msgs/Octomap` | OctoMap for collision checking |
| **Pub** | `/command/trajectory` | `trajectory_msgs/MultiDOFJointTrajectory` | Trajectory waypoints for the controller |
| **Pub** | `/rrt_status` | `std_msgs/String` | Status messages (`"reached"`, `"planning"`, `"failed"`) |
| **Pub** | `/path` | `nav_msgs/Path` | Full planned path (for visualization) |
| **Pub** | `/rrt_tree` | `visualization_msgs/MarkerArray` | RRT tree nodes and edges |
| **Pub** | `/rrt_path` | `visualization_msgs/Marker` | Final RRT path (green line) |
| **Pub** | `/rrt_waypoint_path` | `visualization_msgs/MarkerArray` | Downsampled waypoint path (orange) |

---

## ROS Parameters

### Global Planner Node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `home_x` | `-360.0` | Home X coordinate for return-to-home |
| `home_y` | `0.0` | Home Y coordinate |
| `home_z` | `16.0` | Home Z coordinate (altitude) |
| `junction_detect_angle_deg` | `60.0` | Maximum angle (°) for clustering frontier directions into one arm |
| `junction_proximity_m` | `30.0` | Minimum distance between confirmed junctions |
| `candidate_merge_radius_m` | `30.0` | Radius for merging junction candidates into existing candidates |
| `no_frontier_timeout_sec` | `12.0` | Seconds without valid frontiers before triggering backtrack |
| `min_gain_threshold` | `0.15` | Minimum exploration gain for a junction arm to be considered unexplored |
| `goal_reached_tolerance_m` | `8.0` | Distance threshold for considering a waypoint reached |
| `exploration_eval_radius_m` | `10.0` | Radius for evaluating exploration gain around a point |
| `stagnation_timeout_sec` | `20.0` | Seconds of map stagnation before triggering backtrack |
| `min_volume_delta` | `5` | Minimum OctoMap leaf-node growth to not be considered stagnant |
| `junction_confirm_count` | `0` | Number of detections required to confirm a junction candidate (0 = immediate) |
| `min_frontier_gain_for_timer` | `0.3` | Minimum frontier gain to reset the no-frontier timer |
| `breadcrumb_spacing_m` | `30.0` | Distance between recorded breadcrumb positions |

### Local Planner Node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_radius_` | `10.0` | Bounding box half-size for volumetric gain calculation |
| `search_radius_` | `60.0` | Maximum distance from drone to consider OctoMap leaves |
| `x_limit_position_` | `-370.0` | X-axis boundary limit for frontier search |
| `min_publish_gain_` | `1.3` | Minimum `volumetric × direction` gain to publish a frontier |
| `occupied_threshold_` | `0.5` | OctoMap occupancy threshold for "occupied" classification |

### RRT 3D Planner Node

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_size` | `2.0` | RRT extension step size (m) |
| `goal_tolerance` | `2.0` | Distance to goal to consider path complete (m) |
| `node_limit` | `5000` | Maximum RRT tree nodes per planning attempt |
| `safety_radius_` | `4.0` | Spherical collision-check radius (m) |
| `occupied_threshold_` | `0.5` | OctoMap occupancy threshold |
| `rrt_timeout_` | `5.0` | Maximum planning time per attempt (s) |
| `desired_velocity` | `15.0` | Desired flight velocity for trajectory timing (m/s) |

---

## RViz Visualization

The system publishes extensive RViz markers for debugging and monitoring:

### Local Planner Markers (`/frontier_goals_markers`)

- **Green cubes**: Frontier candidate positions
- **Text labels**: Volumetric gain value above each frontier

### Global Planner Markers (`/global_frontier_markers`)

| Marker | Shape | Color | Description |
|--------|-------|-------|-------------|
| Junction (unexplored) | Sphere (r=3) | Cyan | Junction with unexplored arms remaining |
| Junction (explored) | Sphere (r=3) | Gray | Fully explored junction |
| Arm (unexplored) | Arrow | Green | Unexplored junction arm |
| Arm (explored) | Arrow | Gray | Already-explored arm |
| Arm (target) | Arrow | Orange | Currently targeted arm |
| Arm labels | Text | White | Arm gain values |
| Junction labels | Text | White | Junction ID and unexplored count |
| Home position | Cube (2m) | Gold | Return-to-home target |
| State label | Text | White | Current FSM state |

### RRT Planner Markers

| Topic | Description |
|-------|-------------|
| `/rrt_tree` | Blue spheres (nodes) + white lines (edges) of the RRT tree |
| `/rrt_path` | Green line strip — the final planned path |
| `/rrt_waypoint_path` | Orange line + spheres — downsampled waypoints sent to the controller |

---

## Dependencies

### System Dependencies

- **ROS 2 Humble**
- **OctoMap** (`octomap`, `octomap_msgs`)
- **Eigen3**

### ROS Package Dependencies

| Package | Used By |
|---------|---------|
| `rclcpp` | All nodes |
| `geometry_msgs` | All nodes |
| `nav_msgs` | All nodes |
| `octomap_msgs` / `octomap` | All nodes (3D occupancy) |
| `visualization_msgs` | All nodes (RViz markers) |
| `trajectory_msgs` | Local Planner, RRT |
| `std_msgs` | Global Planner, RRT |
| `tf2` / `tf2_eigen` / `tf2_geometry_msgs` | Local Planner |
| `mav_msgs` / `mav_planning_msgs` | Local Planner (package deps) |
| `Eigen3` | Local Planner, Global Planner |

---

## Building & Running

### Build

From the ROS 2 workspace root:

```bash
cd ~/dev/autonomous-systems-team-1/ros2_ws

# Build both packages
colcon build --packages-select planner_pkg multidim_rrt_cpp

# Source the workspace
source install/setup.bash
```

### Run Individual Nodes

```bash
# Terminal 1 – Local Planner
ros2 run planner_pkg local_planner_node

# Terminal 2 – Global Planner
ros2 run planner_pkg global_planner_node

# Terminal 3 – RRT 3D Path Planner
ros2 run multidim_rrt_cpp rrt3D
```

### Override Parameters at Launch

```bash
ros2 run planner_pkg global_planner_node --ros-args \
  -p home_x:=-360.0 \
  -p home_y:=0.0 \
  -p home_z:=16.0 \
  -p no_frontier_timeout_sec:=15.0 \
  -p breadcrumb_spacing_m:=25.0
```

### Monitor Exploration State

```bash
# Watch the current exploration state
ros2 topic echo /exploration_state

# Watch RRT planning status
ros2 topic echo /rrt_status
```
