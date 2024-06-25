#!/bin/bash

# Get the current directory
dir_path=$(pwd)

# Create named pipes
mkfifo /tmp/shared_input1
mkfifo /tmp/shared_input2

# Command to run ROS Noetic in terminal 0
cmd0="source /opt/ros/noetic/setup.bash; roscore"

# Command to be executed in each terminal, including 'conda init'
cmd1="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; $HOME/miniconda3/envs/franka/bin/python3 urRobot/mainUR5_v2.py < /tmp/shared_input1"
cmd2="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; $HOME/miniconda3/envs/franka/bin/python dataLabeling/digitalGloveNodeUSB.py"
cmd3="source $(conda info --base)/etc/profile.d/conda.sh; conda init bash; conda activate frankapyenv; source /opt/ros/noetic/setup.bash; $HOME/miniconda3/envs/franka/bin/python3 urRobot/saveDataNode.py < /tmp/shared_input2"

# Open new terminals with specific titles and run the commands
gnome-terminal --working-directory="$dir_path" --title="ROS Noetic" -- bash -c "$cmd0; exec bash" &
pid0=$!
echo "PID of terminal 0: $pid0"

gnome-terminal --working-directory="$dir_path" --title="Main UR5" -- bash -c "$cmd1; exec bash" &
pid1=$!

gnome-terminal --working-directory="$dir_path" --title="Digital Glove Node" -- bash -c "$cmd2; exec bash" &
pid2=$!

gnome-terminal --working-directory="$dir_path" --title="Save Data Node" -- bash -c "$cmd3; exec bash" &
pid3=$!
sleep 1

# Arrange the terminals using wmctrl
wmctrl -r "ROS Noetic" -e 0,0,0,800,600
wmctrl -r "Main UR5" -e 0,800,0,800,600
wmctrl -r "Digital Glove Node" -e 0,0,600,800,600
wmctrl -r "Save Data Node" -e 0,800,600,800,600

# Function to stop all terminals
stop_terminals() {
    kill $pid0 $pid1 $pid2 $pid3
    echo "Terminals stopped and closed."
}

# Arrange the terminals using wmctrl
wmctrl -r "ROS Noetic" -e 0,0,0,800,600
wmctrl -r "Main UR5" -e 0,800,0,800,600
wmctrl -r "Digital Glove Node" -e 0,0,600,800,600
wmctrl -r "Save Data Node" -e 0,800,600,800,600

# Trap function to call stop_terminals on exit
trap stop_terminals EXIT

# Read input from the main terminal and broadcast it to the named pipes using tee
echo "Enter the shared input: "
cat | tee /tmp/shared_input1 > /tmp/shared_input2

# Clean up
rm /tmp/shared_input1 /tmp/shared_input2
