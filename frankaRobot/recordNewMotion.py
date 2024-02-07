'''
By Maryam Rezayati

How to run?
## 1st  Step: unlock robot
	1) connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	2) unlock the robot and activate FCI

## 2nd Step: run frankapy
   -open an terminal
	conda activate frankapyenv
	bash /home/mindlab/franka/run_frankapy.sh

## 3nd Step: run this code
    -open another terminal 
	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
	source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend

	/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/contactInterpretation/frankRobot/recordNewMotion.py

'''
import numpy as np
import time
import pandas as pd
from frankapy import FrankaArm
from threading import Thread

main_path = '/home/mindlab/contactIntrpretation/frankaRobot/'

def move_robot(fa: FrankaArm, duration: float):
    print('Thread_move_robot has started.', duration)
    try:
        fa.run_guide_mode(duration=duration)
    except Exception as e:
        print(e)
    print('Exit thread!')

if __name__ == '__main__':
    # Initialize FrankaArm
    fa = FrankaArm()

    # Get user input for the duration of robot motion
    duration = float(input('How long does it take to put the robot in the desired positions? (sec)   '))

    # Create a thread for robot motion
    thread_move_robot = Thread(target=move_robot, args=[fa, duration])

    # Lists to store joint data
    joints = []

    # Prompt user to start the process
    state = input('Are you ready to start? (yes/no)   ')

    if state == 'yes':
        # Start the thread for robot motion
        thread_move_robot.start()
        start_time = time.time()
        counter = 0

        # Manual input of desired robot positions
        while (time.time() - start_time) < duration:
            state = input('Press enter when the robot is in a desired position:   ')
            dummy = fa.get_joints()
            dummy[6] += np.deg2rad(45)
            print('\n', dummy)
            joints.append(dummy)
            counter += 1
            print(counter, 'poses are selected. All positions will be saved in a file when this program ends.\n')

        # Create a DataFrame and save joint data to a CSV file
        df = pd.DataFrame(data=np.squeeze(joints))
        named_tuple = time.localtime()
        filename = main_path + 'robotMotionJointData' + str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple)) + '.csv'
        df.to_csv(filename)
        print('Data are saved in', filename)

        # End the program
        exit()
