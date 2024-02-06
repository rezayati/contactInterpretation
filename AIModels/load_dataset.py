import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class create_tensor_dataset(Dataset):
    def __init__(self, path = './dataset/realData/contact_detection_train.csv', transform = transforms.Compose([transforms.ToTensor()]), num_classes =5,
                 num_features_dataset = 28, num_features_lstm = 4, data_seq = 28, desired_seq = 28, localization= False, collision =False):
        self.path = path
        self.transform = transform
        self.num_features_dataset = num_features_dataset
        self.num_features_lstm = num_features_lstm
        self.data_seq = data_seq
        self.desired_seq = desired_seq
        self.dof = 7
        self.num_classes = num_classes
        self.localization = localization
        self.collision = collision
        if collision and localization:
            print('collision and localization cannot be true at the same time!')
            exit()
            

        self.read_dataset()
        self.data_in_seq()
        
    def __len__(self):
        return len(self.data_target)


    def __getitem__(self, idx: int):

        data_sample = torch.tensor(self.data_input.iloc[idx].values)
        data_sample = torch.reshape(data_sample, (self.dof ,self.num_features_lstm*self.desired_seq))

        target = self.data_target.iloc[idx]

        return data_sample, target


    def read_dataset(self):
        
        labels_map = {
            0: 'Noncontact',
            1: 'Intentional_Link5',
            2: 'Intentional_Link6',
            3: 'Collision_Link5',
            4: 'Collision_Link6',
        }
        # laod data from csv file
        data = pd.read_csv(self.path)
        # specifying target and data
        data_input = data.iloc[:,1:data.shape[1]]
        data_target = data['Var1']

        # changing labels to numbers
        for i in range(data_input.shape[0]):
            for j in range(len(labels_map)):

                if self.localization:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==3:
                            data_target.iat[i] = 0
                        elif j==2 or j==4:
                            data_target.iat[i] = 1

                elif self.collision:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==2:
                            data_target.iat[i] = 0
                        elif j==3 or j==4:
                            data_target.iat[i] = 1

                elif self.num_classes == 3 or self.collision:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0
                        elif j==1 or j==2:
                            data_target.iat[i] = 1
                        elif j==3 or j==4:
                            data_target.iat[i] = 2

                elif self.num_classes == 5:
                    if data.iloc[i, 0] == labels_map[j]:
                        data_target.iat[i] = j

                elif self.num_classes ==2:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0
                        else:
                            data_target.iat[i] = 1

                else: 
                    print('ERROR! num_classes should be 2 or 3 or 5')
                    exit()

        if self.localization or self.collision:
            data_input = data_input[data_target.iloc[:]!=-1]
            data_target = data_target[data_target.iloc[:]!=-1]

        self.data_input = data_input.reset_index(drop=True)
        self.data_target = data_target.reset_index(drop=True)

    def data_in_seq(self):

        dof = self.dof

        # resorting item position
        data = np.array( range(0, self.num_features_dataset * self.data_seq ))
        data = data.reshape(self.data_seq, self.num_features_dataset)

        joint_data_pos = []
        for j in range(dof):
            # (4,28) : [tau(t), tau_ext(t), e(t), de(t)]j
            join_data_matrix = data[:, [j, j+dof, j+dof*2, j+dof*3 ]]
            joint_data_pos.append(join_data_matrix.reshape((4*28)))
        
        joint_data_pos = np.hstack(joint_data_pos)

        # resorting (28,28)---> (4,28)(4,28)(4,28)(4,28)(4,28)(4,28)(4,28)

        self.data_input.columns = range(self.num_features_dataset * self.data_seq)
        self.data_input = self.data_input.loc[:][joint_data_pos]


#it is like the create_tensor_dataset, but with this function we can select the features in the dataset.
class create_tensor_dataset_without_torque(Dataset):
    
    def __init__(self, path = './dataset/realData/contact_detection_train.csv', transform = transforms.Compose([transforms.ToTensor()]), num_classes =5,
                 num_features_dataset = 28, num_features_lstm = 4, data_seq = 28, desired_seq = 28, localization= False, collision =False):
        self.path = path
        self.transform = transform
        self.num_features_dataset = num_features_dataset
        self.num_features_lstm = num_features_lstm
        self.data_seq = data_seq
        self.desired_seq = desired_seq
        self.dof = 7
        self.num_classes = num_classes
        self.localization = localization
        self.collision = collision
        if collision and localization:
            print('collision and localization cannot be true at the same time!')
            exit()
            

        self.read_dataset()
        self.data_in_seq()
        
    def __len__(self):
        return len(self.data_target)


    def __getitem__(self, idx: int):

        data_sample = torch.tensor(self.data_input.iloc[idx].values)
        data_sample = torch.reshape(data_sample, (self.dof ,self.num_features_lstm*self.desired_seq))

        target = self.data_target.iloc[idx]

        return data_sample, target


    def read_dataset(self):
        
        labels_map = {
            0: 'Noncontact',
            1: 'Intentional_Link5',
            2: 'Intentional_Link6',
            3: 'Collision_Link5',
            4: 'Collision_Link6',
        }
        # laod data from csv file
        data = pd.read_csv(self.path)
        # specifying target and data
        data_input = data.iloc[:,1:data.shape[1]]
        data_target = data['Var1']

        # changing labels to numbers
        for i in range(data_input.shape[0]):
            for j in range(len(labels_map)):

                if self.localization:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==3:
                            data_target.iat[i] = 0
                        elif j==2 or j==4:
                            data_target.iat[i] = 1

                elif self.collision:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = -1
                        elif j==1 or j==2:
                            data_target.iat[i] = 0
                        elif j==3 or j==4:
                            data_target.iat[i] = 1

                elif self.num_classes == 3 or self.collision:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0
                        elif j==1 or j==2:
                            data_target.iat[i] = 1
                        elif j==3 or j==4:
                            data_target.iat[i] = 2

                elif self.num_classes == 5:
                    if data.iloc[i, 0] == labels_map[j]:
                        data_target.iat[i] = j

                elif self.num_classes ==2:
                    if data.iloc[i, 0] == labels_map[j]:
                        if j==0:
                            data_target.iat[i] = 0
                        else:
                            data_target.iat[i] = 1

                else: 
                    print('ERROR! num_classes should be 2 or 3 or 5')
                    exit()

        if self.localization or self.collision:
            data_input = data_input[data_target.iloc[:]!=-1]
            data_target = data_target[data_target.iloc[:]!=-1]

        self.data_input = data_input.reset_index(drop=True)
        self.data_target = data_target.reset_index(drop=True)

    def data_in_seq(self):

        dof = self.dof

        # resorting item position
        data = np.array( range(0, self.num_features_dataset * self.data_seq ))
        data = data.reshape(self.data_seq, self.num_features_dataset)

        joint_data_pos = []
        for j in range(dof):
            # (4,28) : [tau(t), tau_ext(t), e(t), de(t)]j
            if self.num_features_lstm == 4:
                column_index = [j, j+dof, j+dof*2, j+dof*3 ]
            elif self.num_features_lstm == 2:
                column_index = [j+dof*2, j+dof*3 ]
            
            elif self.num_features_lstm == 3:
                column_index = [j+dof, j+dof*2, j+dof*3 ]
                 
                
            row_index= range(self.data_seq-self.desired_seq, self.data_seq)
            join_data_matrix = data[:, column_index]

            joint_data_pos.append(join_data_matrix.reshape((len(column_index)*len(row_index))))
        
        joint_data_pos = np.hstack(joint_data_pos)

        # resorting (28,28)---> (4,28)(4,28)(4,28)(4,28)(4,28)(4,28)(4,28)

        self.data_input.columns = range(self.num_features_dataset * self.data_seq)
        self.data_input = self.data_input.loc[:][joint_data_pos]

# read data can read fron pickle. it is for the new dataset which has less complexity in data_in_seq method.
class create_tensor_dataset_localization(Dataset):
    def __init__(self, path = './dataset/realData/contact_detection_train.csv', transform = transforms.Compose([transforms.ToTensor()]), num_classes =5,
                 num_features_dataset = 14, num_features_lstm = 2, data_seq = 28, desired_seq = 28, localization= False, collision =False):
        self.path = path
        self.transform = transform
        self.num_features_dataset = num_features_dataset
        self.num_features_lstm = num_features_lstm
        self.data_seq = data_seq
        self.desired_seq = desired_seq
        self.dof = 7
        self.num_classes = num_classes
        self.localization = localization
        self.collision = collision
        if collision and localization:
            print('collision and localization cannot be true at the same time!')
            exit()
            

        self.read_dataset()
        self.data_in_seq()
        
    def __len__(self):
        return len(self.data_target)


    def __getitem__(self, idx: int):

        data_sample = torch.tensor(self.data_input.iloc[idx].values)
        data_sample = torch.reshape(data_sample, (self.dof ,self.num_features_lstm*self.desired_seq))

        target = self.data_target.iloc[idx]

        return data_sample, target


    def read_dataset(self):
        
        # laod data from csv file
        if self.path[(len(self.path)-3): len(self.path)] == 'csv':
            data = pd.read_csv(self.path)
        elif self.path[(len(self.path)-3): len(self.path)] == 'pkl':
            data = pd.read_pickle(self.path)
        # specifying target and data
        data_input = data.iloc[:,1:data.shape[1]]
        data_target = data.iloc[:,0]

        if not self.localization:
            data_target.loc[data_target.iloc[:]!=0] = 1

        if self.localization or self.collision:
            data_input = data_input.loc[data_target.iloc[:]!=0, :]
            data_target = data_target.loc[data_target.iloc[:]!=0]
            data_target = data_target-1


        self.data_input = data_input.reset_index(drop=True)
        self.data_target = data_target.reset_index(drop=True)
        

    def data_in_seq(self):

        dof = self.dof

        # resorting item position
        data = np.array( range(0, self.num_features_dataset * self.data_seq ))
        data = data.reshape(self.data_seq, self.num_features_dataset)

        joint_data_pos = []
        for j in range(dof):
                 
            column_index = np.array(range(self.num_features_lstm))*dof +j
            row_index= range(self.data_seq-self.desired_seq, self.data_seq)
            join_data_matrix = data[:, column_index]
            joint_data_pos.append(join_data_matrix.reshape((len(column_index)*len(row_index))))
        
        joint_data_pos = np.hstack(joint_data_pos)

        # resorting (28,28)---> (4,28)(4,28)(4,28)(4,28)(4,28)(4,28)(4,28)

        self.data_input.columns = range(self.num_features_dataset * self.data_seq)
        self.data_input = self.data_input.loc[:][joint_data_pos]
