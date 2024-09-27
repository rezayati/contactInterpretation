import sys
import os
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import time
# Load the model from file
pipe = joblib.load('AIModels/model_pipeline.pkl')

#bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost

# Source ROS setup.bash
os.system('source /opt/ros/noetic/setup.bash')

# Source your Franka workspace
os.system('source /home/rzma/robotsAPI/franka-interface/catkin_ws/devel/setup.bash')
os.system('source /home/rzma/robotsAPI/frankapy/catkin_ws/devel/setup.bash')

# Add ROS Python libraries to Python path
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

# Manually add your workspace to ROS_PACKAGE_PATH
sys.path.append('/home/rzma/robotsAPI/franka-interface/catkin_ws/devel/lib/python3/dist-packages')

import rospy
from franka_interface_msgs.msg import RobotState


# Define global variables
global window
window = np.zeros((28, 7))
features_num = 7
num_features_lstm = 1
dof = 7

def contact_detection(data):
    global window
    start_time=time.time()
    e_q = np.array(data.q_d) - np.array(data.q)
    
    new_row = e_q.reshape((1, features_num))
    window = np.append(window[1:,:], new_row, axis=0)

    stackedWindow = np.hstack(window).reshape(1, 196)
    
    # Assuming the 'pipe' object is properly defined
    model_out = pipe.predict(stackedWindow)
    print(time.time()-start_time, model_out)

if not rospy.core.is_initialized():
    rospy.init_node('contact_detection_node', anonymous=True)
    print('Node initialized')
else:
    print('Node was already initialized')


# Set up the subscriber
rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState, callback=contact_detection)
print('Subscriber initialized')  # Print to confirm the subscriber is active

# Keep the node alive and spinning
rospy.spin()
