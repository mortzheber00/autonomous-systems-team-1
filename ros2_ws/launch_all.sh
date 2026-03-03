#!/bin/bash

# Source the ROS2 workspace
source "$(dirname "$0")/install/setup.bash"

echo "Launching simulation..."
ros2 launch simulation simulation.launch.py &
SIM_PID=$!

echo "Waiting 10 seconds for simulation to initialize..."
sleep 10

echo "Launching state machine..."
ros2 launch state_machine_pkg state_machine.launch.py &
SM_PID=$!

# Wait for both processes, exit if either dies
wait $SIM_PID $SM_PID
