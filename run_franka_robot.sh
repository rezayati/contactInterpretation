#!/bin/bash

# Get the current directory
dir_path=$(pwd)

# Create named pipes
mkfifo /tmp/shared_input2

# Command to run ROS Noetic in terminal 0
cmd0="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost"
sleep 2

# Command to be executed in each terminal, including 'conda init'
cmd1="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; \
source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend; source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend; \
$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/main.py"

cmd2="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; \
$HOME/miniconda/envs/frankapyenv/bin/python3 dataLabeling/digitalGloveNodeUSB.py"

cmd3="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; \
source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend; source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend; \
$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/saveDataNode.py < /tmp/shared_input2"

# Open new terminals with specific titles and run the commands
gnome-terminal --working-directory="$dir_path" --title="ROS Noetic" -- bash -c "$cmd0; exec bash" &
pid0=$!
echo "PID of terminal 0: $pid0"

gnome-terminal --working-directory="$dir_path" --title="Main UR5" -- bash -c "$cmd1; exec bash" &
pid1=$!
echo "PID of terminal 1: $pid1"

gnome-terminal --working-directory="$dir_path" --title="Digital Glove Node" -- bash -c "$cmd2; exec bash" &
pid2=$!
echo "PID of terminal 2: $pid2"

gnome-terminal --working-directory="$dir_path" --title="Save Data Node" -- bash -c "$cmd3; exec bash" &
pid3=$!
echo "PID of terminal 3: $pid3"

# Arrange the terminals using wmctrl
wmctrl -r "ROS Noetic" -e 0,0,0,800,600
wmctrl -r "Main UR5" -e 0,800,0,800,600
wmctrl -r "Digital Glove Node" -e 0,0,600,800,600
wmctrl -r "Save Data Node" -e 0,800,600,800,600

# Read input from the main terminal and broadcast it to the named pipes using tee
echo "Enter the shared input: "
cat | tee /tmp/shared_input2

# Clean up named pipes on exit
rm /tmp/shared_input2
exit
