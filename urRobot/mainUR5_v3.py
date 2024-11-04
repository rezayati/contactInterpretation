
'''
conda activate frankapyenv
source /opt/ros/noetic/setup.bash
'''
import numpy as np
import rospy
import torch
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from importModel import import_lstm_models
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import signal
import torch.nn.functional as F

# Robot and Model configurations
robot_ip = '192.168.15.10'
frequency = 200
window_length = 28
dof = 6
features_num = dof * 4
running = True

# Load model
contact_detection_path = '/home/rzma/myProjects/contactInterpretation/AIModels/trainedModels/contactDetection/trainedModel_06_30_2023_10:16:53.pth'
contact_detection_path = '/home/rzma/myProjects/contactInterpretation/AIModels/trainedModels/contactDetection/trainedModel_01_24_2024_11:18:01.pth'
model_contact, _, num_features_lstm = import_lstm_models(contact_detection_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_contact = model_contact.to(device)

# Data window for contact detection
window = np.zeros((window_length, features_num))
k = np.array([0.1082, 0.1100, 0.1097, 0.0787, 0.0294, 0.0261]) * 13

def signal_handler(sig, frame):
    running = False
    print("Exiting program...")
    

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    rospy.init_node("robot_contact_detection")
    data_object = RTDEReceive(robot_ip, frequency)

    while not rospy.is_shutdown() and running and data_object.isConnected():
        try:
            # Retrieve robot data
            q_desired = np.array(data_object.getTargetQ())
            q = np.array(data_object.getActualQ())
            dq_desired = np.array(data_object.getTargetQd())
            dq = np.array(data_object.getActualQd())
            i_desired = np.array(data_object.getTargetCurrent())
            i_actual = np.array(data_object.getActualCurrent())
            tau_J = np.array(data_object.getTargetMoment())
            tau_ext = np.array(i_desired - i_actual)
            tau_ext = np.multiply(tau_ext, k)

            e_q = q_desired - q
            e_dq = dq_desired - dq

            new_row = np.hstack((tau_J, tau_ext, e_q, e_dq)).reshape((1, features_num))
            window = np.append(window[1:], new_row, axis=0)

            lstmDataWindow = []
            for j in range(dof):
                if num_features_lstm == 4:
                    column_index = [j, j + dof, j + dof * 2, j + dof * 3]
                elif num_features_lstm == 2:
                    column_index = [j + dof * 2, j + dof * 3]
                elif num_features_lstm == 3:
                    column_index = [j + dof, j + dof * 2, j + dof * 3]

                join_data_matrix = window[:, column_index]
                lstmDataWindow.append(join_data_matrix.reshape((1, num_features_lstm * window_length)))

            lstmDataWindow = np.vstack(lstmDataWindow)

            data_input = torch.Tensor(np.array([lstmDataWindow])).to(device).float()

            # Run contact detection model
            with torch.no_grad():
                #contact = model_contact(data_input).argmax().item()
                contact = model_contact(data_input)
                probabilities = F.softmax(contact, dim=1)  # dim=1 for column-wise softmax
            
            #contact = probabilities.argmax(dim=1).item()
                
            rospy.loginfo(probabilities)

            '''if contact == 1:
                rospy.loginfo("Contact detected!")
            else:
                rospy.loginfo("No contact detected.")'''

        except Exception as e:
            rospy.logerr(f"Error in contact detection: {e}")
            running = False

    # Clean up resources
    data_object.disconnect()
    print("Program stopped.")
