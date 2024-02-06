'''
By Maryam Rezayati

How to run?
## 1st  Step: unlock robot
	1) connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	2) unlock the robot and activate FCI

## 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	cd /home/mindlab/franka
	bash run_frankapy.sh

## 3nd Step: run this code

open another terminal 

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

main_path= '/home/mindlab/contactIntrpretation/frankaRobot/'

def move_robot(fa: FrankaArm, duration: float):
    print('thread_move_robot is started ',duration)
    try:
        fa.run_guide_mode( duration= duration )
    except Exception as e:
        print(e)
    print('exit thread!')

if __name__ == '__main__':
    fa = FrankaArm()
    duration = input('How lond does it take to put the robot in the desired positions? (sec)   ')
    duration = float(duration)

    thread_move_robot = Thread(target= move_robot, args=[fa,duration])
    
    
    joints = []
    state = 'no'
    while(state!='yes'):
        
        state = input('are you ready to start? (yes/no)   ')
        if state == 'yes':
            thread_move_robot.start()
            start_time = time.time()
            counter = 0
            while((time.time()-start_time) < int(duration)):
                state = input('press enter when robot is in a desired position:   ')
                dummy = fa.get_joints()
                #dummy = np.zeros(7)
                dummy[6] += np.deg2rad(45)
                print('\n',dummy)
                joints.append(dummy)
                counter += 1
                print(counter, 'poses are selected. All positions will be saved in a file when this program ends. \n')
            df = pd.DataFrame(data=np.squeeze(joints))
            named_tuple = time.localtime() 

            df.to_csv(main_path+'robotMotionJointData'+str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple))+'.csv')
            print('data are saved in ', main_path+'robotMotionJointData'+str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple))+'.csv')
            exit()