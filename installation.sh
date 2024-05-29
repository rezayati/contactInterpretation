#!/bin/bash
# 
# Make the Script Executable: chmod +x install_franka_env.sh
# Run the script: ./installation.sh

set -e

# Define variables
CONDA_ENV_NAME="frankapyenv"
REQUIREMENTS_FILE="requirements.txt"
MINICONDA_VERSION="Miniconda3-py38_4.9.2-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_VERSION"
PYTHON_VERSION="3.9"
FRANKAPY_REPO="https://github.com/iamlab-cmu/FrankaPy"
FRANKAPY_VERSION="v0.3.0"
FRANKA_INTERFACE_REPO="https://github.com/iamlab-cmu/franka-interface"
ROS_VERSION="noetic"
LIBFRANKA_VERSION=5

# Notify user about the installation
echo "  "
echo "****    please install a real-time kernel first and then run this installation **** "
echo "  "
echo "This script will install the following components:"
echo "- Miniconda"
echo "- Conda environment (${CONDA_ENV_NAME}) with Python ${PYTHON_VERSION}"
echo "- ROS ${ROS_VERSION}"
echo "- Franka-interface"
echo "- FrankaPy"
echo ""
read -p "Do you want to proceed with the installation? (yes/no): " response

if [[ "$response" != "yes" ]]; then
    echo "Installation aborted by user."
    exit 1
fi

# Update and install necessary system dependencies
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    wget \
    software-properties-common \
    build-essential \
    cmake \
    git

# Install Miniconda
if ! command -v conda &> /dev/null
then
    echo "Installing Miniconda..."
    wget $MINICONDA_URL -O $MINICONDA_VERSION
    bash $MINICONDA_VERSION -b -p $HOME/miniconda
    rm $MINICONDA_VERSION
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "Conda is already installed"
fi

# Create and activate the Conda environment
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "Conda environment ${CONDA_ENV_NAME} already exists"
else
    echo "Creating Conda environment ${CONDA_ENV_NAME} with Python ${PYTHON_VERSION}..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
fi
source $HOME/miniconda/bin/activate $CONDA_ENV_NAME

#install requirement file
pip install -r requirement.txt
# Install ROS $ROS_VERSION
if ! [ -x "$(command -v roscore)" ]; then
    echo "Installing ROS $ROS_VERSION..."
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y ros-$ROS_VERSION-desktop-full
    echo "source /opt/ros/$ROS_VERSION/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    sudo apt install -y python3-rosdep
    sudo rosdep init
    rosdep update
else
    echo "ROS $ROS_VERSIONic is already installed"
fi

# Install Franka-interface dependencies
echo "Installing Franka-interface dependencies..."
sudo apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    libpoco-dev \
    libcurl4-openssl-dev \
    libtinyxml2-dev \
    ros-$ROS_VERSION-ros-control \
    ros-$ROS_VERSION-ros-controllers \
    ros-$ROS_VERSION-moveit\
    ros-$ROS_VERSION-libfranka\
    ros-$ROS_VERSION-franka-ros
    
mkdir -p robotAPI
cd robotAPI
# Clone and install Franka-interface
if [ ! -d "franka-interface" ]; then
    echo "Cloning Franka-interface..."
    git clone --recurse-submodules $FRANKA_INTERFACE_REPO
    cd franka-interface
    # Clone the appropriate version of libfranka
    echo "Cloning LibFranka version $LIBFRANKA_VERSION..."
    bash ./bash_scripts/clone_libfranka.sh $LIBFRANKA_VERSION

    # Build libfranka
    echo "Building LibFranka..."
    bash ./bash_scripts/make_libfranka.sh

    # Build franka-interface
    echo "Building Franka-interface..."
    bash ./bash_scripts/make_franka_interface.sh

    # Install Python dependencies for catkin build
    echo "Installing catkin build dependencies..."
    pip install catkin-tools empy==3.3.4

    # Build the catkin workspace
    echo "Building the catkin workspace..."
    bash ./bash_scripts/make_catkin.sh
    
    # Source the catkin workspace setup script
    source catkin_ws/devel/setup.bash
    cd ..
else
    echo "Franka-interface already cloned"
fi



# Clone and install FrankaPy
if [ ! -d "frankapy" ]; then
    echo "Cloning FrankaPy ...."
    git clone --recurse-submodules $FRANKAPY_REPO "frankapy"
    cd "frankapy"

    # Install Python dependencies for FrankaPy
    echo "Installing Python dependencies for FrankaPy..."
    pip install -e .

    # Build the catkin workspace
    echo "Building the catkin workspace..."
    ./bash_scripts/make_catkin.sh

else
    echo "FrankaPy already cloned in $FRANKAPY_DIR"
    cd $FRANKAPY_DIR
fi


echo "Installation complete. Don't forget to source your ROS environment:"
echo "source /opt/ros/$ROS_VERSION/setup.bash"
echo "And activate your Conda environment with:"
echo "conda activate ${CONDA_ENV_NAME}"
echo " "
echo " go to robotAPI/frankapy/bash_scripts/start_control_pc.sh and change the setting.
