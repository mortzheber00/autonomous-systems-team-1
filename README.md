# AutonomousSystems2025


## SubTerrain Challenge (Project)

![Schematics of the Drone](DroneforCave.png)



## Setup
```
cd ros2_ws
colcon build
source install/setup.bash
```


## Simulation
Copy the unpacked simulation files into the folder `ros2_ws/install/simulation/lib/simulation`.
Make `Simulation.x86_64` executable.

Run:

```
ros2 launch simulation simulation.launch.py
```