#!/usr/bin/env python3
"""
By Maryam Rezayat

This code continuously records data from the Franka Panda robot until manually stopped. It prompts the user to enter a tag name for labeling the data.

Files Created:
1. all_data.txt: Contains all data published by the Franka Panda robot.
2. true_label.csv: Contains data of true labels acquired through CANBUS communication.
3. model_result.csv: Presents the model output and data window.

How to Run:
  - open a terminal
    conda activate frankapyenv
    source /opt/ros/noetic/setup.bash
    $HOME/miniconda/envs/frankapyenv/bin/python3 urRobot/saveDataNode.py
"""

import numpy as np
import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import csv
import os
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
from threading import Thread

# Set the default PATH
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/urRobot/DATA/model_out/'
robot_dof = 6
num_features = 4 * robot_dof
time_seq = 28
num_columns = time_seq * num_features

class LogData:
    def __init__(self, PATH: str, folder_name: str) -> None:
        # Initialize ROS node
        rospy.init_node('log_data')
        rospy.on_shutdown(self.shutdown_hook)
        self.running = True

        # Create a folder for saving data
        self.PATH = PATH + folder_name
        os.makedirs(self.PATH, exist_ok=True)
        print("Directory '%s' created" % self.PATH)

        # Create empty files for saving data
        header = ['Time_sec', 'Time_nsec', 'prediction_duration', 'contact_out', 'collision_out', 'localization_out'] + [f'val_{i}' for i in range(num_columns)]
        self.log_model_result = csv.writer(open(self.PATH + '/model_out.csv', 'w'))
        self.log_model_result.writerow(header)
        self.file_index = csv.writer(open(self.PATH + '/true_label.csv', 'w'))
        self.file_index.writerow(('time_sec', 'time_nsec', 'timestamp', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4'))
        print('*** empty text files are created in ', self.PATH, ' ***')

        # Subscribe to relevant ROS topics
        rospy.Subscriber(name="/contactTimeIndex", data_class=numpy_msg(Floats), callback=self.save_contact_index)

        # Start a separate thread to save robot data
        self.robot_data_thread = Thread(target=self.save_robot_data)
        self.robot_data_thread.start()

    def shutdown_hook(self):
        # Set running flag to False to allow clean exit
        self.running = False
        rospy.loginfo("Shutdown signal received. Stopping data recording.")

    def save_robot_data(self):
        try:
            while self.running and not rospy.is_shutdown():
                rospy.loginfo("init save robot data")
                data_object = RTDEReceive("192.168.15.10", 200)
                file_name = self.PATH + '/' + str(rospy.Time.now().to_sec()) + '.txt'
                data_object.startFileRecording(file_name)

                while self.running and not rospy.is_shutdown():
                    if data_object.isConnected():
                        rospy.sleep(1)
                    else:
                        rospy.logwarn("Robot not connected.")
                        break

        except Exception as e:
            rospy.logerr(f"Error in save_robot_data: {e}")
        finally:
            data_object.stopFileRecording()
            rospy.loginfo("File recording stopped.")

    def save_contact_index(self, data):
        # Save contact index data to true_label.csv
        data_row = np.array(data.data)
        self.file_index.writerow(data_row)

    def save_model_output(self, data):
        # Save model output data to model_result.csv
        data_row = np.array(data.data)
        self.log_model_result.writerow(data_row)

if __name__ == "__main__":
    try:
        # Prompt the user to enter a tag name for data labeling
        folder_name = input('Enter tag name: ')
        
        # Create an instance of the LogData class
        log_data_instance = LogData(main_path, folder_name)
        
        # Keep the program running to listen for ROS messages
        rospy.spin()

    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
    finally:
        log_data_instance.running = False
        rospy.loginfo("Program exited.")
