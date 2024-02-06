#!/usr/bin/env python3
"""
By Maryam Rezayat

The code initially requests a tag name for data labeling, and then continuously records the data until the program is manually stopped.
Executing this code allows you to store all collected data. 
It creates a folder in the specified PATH, and prompts you to enter a tag name for labeling the data. 

    1. all_data.txt : This file encompasses all data published by the Franka Panda robot.
    2. true_label.csv : Contains data of true labels acquired through CANBUS communication.
    3. model_result.csv : Presents the model output and data window.


How to run?
## 1st  Step: Unlock the Robot
1) connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
2) unlock the robot and activate FCI

## 2nd Step: Connecting to the Robot (Runing Frankapy)
open an terminal
cd /home/mindlab/franka
bash run_frankapy.sh

## 3nd Step: Specify the Output Folder (PATH) - (line 56 in the code)

## 4th step: Run the Program
open another terminal:

conda activate frankapyenv
# Source ROS libraries
source /opt/ros/noetic/setup.bash

# Source Franka interface and Frankapy libraries
source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend

/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/contactInterpretation/frankaRobot/saveDataNode.py


"""

## import required libraries 
import numpy as np

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from franka_interface_msgs.msg import RobotState
import csv
import datetime
import os
#publish_output = rospy.Publisher("/showOutput", numpy_msg(Floats), queue_size= 1)

PATH = '/home/mindlab/'


folder_name = input('enter tag name:   ') #str(datetime.datetime.now())a1

class log_data():
    def __init__(self, PATH:str) -> None:
        rospy.init_node('log_data')
        print('ros node is initiated!')
        self.PATH = PATH+folder_name
        os.mkdir(self.PATH)
        print("Directory '% s' created" % self.PATH)

        #creating empty files for saving datag1
        self.file_all = open(self.PATH+'/all_data.txt', 'w')
        self.file_index = csv.writer(open(self.PATH+'/true_label.csv', 'w'))
        self.file_index.writerow(('time_sec', 'time_nsec', 'timestamp', 'DATA0','DATA1', 'DATA2','DATA3','DATA4'))


        self.log_model_reslut = csv.writer(open(self.PATH+'/model_result.csv', 'w'))
        self.log_model_reslut.writerow(('Time_sec','Time_nsec', 'prediction_duration', 'contact_out', 'collision_out', 'localization_out'))
        print('***  Four empty text files are created in ', self.PATH,'  ***')

        '''data = rospy.wait_for_message("/start_time_output", Float64)
        print(data.data)
        self.start_time = np.array(data.data).tolist()
        print(int(self.start_time))
        print(self.start_time - int(self.start_time))'''

        rospy.Subscriber(name= "/model_output",data_class= numpy_msg(Floats), callback =self.save_model_output)
        rospy.Subscriber(name= "/robot_state_publisher_node_1/robot_state",data_class= RobotState, callback= self.save_robot_state)
        rospy.Subscriber(name="/contactTimeIndex", data_class = numpy_msg(Floats), callback = self.save_contact_index)
    
    def save_contact_index(self,data):
        data_row = np.array(data.data)
        #rospy.loginfo(data.data[0])
        self.file_index.writerow(data_row)
        
    def save_model_output(self, data):
        data_row = np.array(data.data)
        self.log_model_reslut.writerow(data_row)
        
    def save_robot_state(self, data):
        #rospy.loginfo("I heard %s", data)
        self.file_all.write(str(data))
                
if __name__ == "__main__":
    #PATH = '/home/rzma/myProjects/sim_to_real/realTimeImplementation/panda_scripts/src/LSTM_DATA/DATA/'
    log_data(PATH)
    rospy.spin()