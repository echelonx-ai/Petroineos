import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Since the data is almost periodic, a neural network should be able to model it pretty easily without the need of smoothing
# or any fancy processing

""" the following dataset loader simply performs the following processing steps:
1. read the data frame
2. normalise it, within the range of -1 to 1 (helps with LSTM networks; since activation is TanH)
3. split into train, val and test splits
4. divide data into sequence windows (i.e. input sequences and corresponding output sequences). This allows easy batching for training
"""

class EnergyDataset(Dataset):
    def __init__(self, args, split): #, target, features, sequence_length=5):
        self.args = args
        self.dataframe =  pd.read_csv(self.args.data_path)
        # normalise data and convert to an array;
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.data_array = self.dataframe['Consumption'].values
        self.data_array = self.scaler.fit_transform(self.data_array.reshape(-1,1))
        self.split = split
        self.input_sequence_length = self.args.input_sequence_length
        self.output_sequence_length = self.args.output_sequence_length

        # split the dataset into training, validation and test set
        train_ratio, val_ratio, test_ratio= 0.7, 0.1, 0.2
        # splitting data into 80-20 i.e. 70% training, 10% val, 20% testing
        if self.split=='train':
            self.df_split= self.data_array[:int(train_ratio * len(self.data_array))]
        elif self.split=='val':
            self.df_split= self.data_array[int(train_ratio * len(self.data_array)):int(train_ratio * len(self.data_array))+int(val_ratio * len(self.data_array))]
        else:
            self.df_split= self.data_array[int((train_ratio+val_ratio) * len(self.data_array)):]

        # divide the data into sliding windows
        # where input and output sequences will be created

        self.df_windows= sliding_windows(self.df_split, self.input_sequence_length, self.output_sequence_length)

        # the model will take as input; input sequence and output; output sequence
    def __getitem__(self, idx): 
        input_arr, gt_arr = self.df_windows[0][idx], self.df_windows[1][idx]
        # convert to tensors;
        input_tensor = torch.tensor(input_arr)
        gt_tensor = torch.tensor(gt_arr)

        return input_tensor, gt_tensor

    def __len__(self):
        return len(self.df_windows[0])


def sliding_windows(data, input_seq_length, output_seq_length=1):
    x = []
    y = []

    for i in range(len(data)-input_seq_length-1):
        _x = data[i:(i+input_seq_length)]
        _y = data[i+input_seq_length:((i+input_seq_length)+output_seq_length)]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


if __name__=='__main__':

    """
    dataset loader test
    """
    import argparse
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_sequence_length', type=int, default=5, help="number of points fed as input into the model")
        parser.add_argument('--output_sequence_length', type=int, default=1, help="number of points outputted by the model")
        parser.add_argument('--data_path', type=str, default='/home/sam37avhvaptuka451/Documents/Contracts/Petroineos/Energy_Demand/energy.dat', help="dir path of where the data is stored")


        args = parser.parse_args()
        return args

    args = get_args()

    db = EnergyDataset(args, 'train')
    print(len(db))

    print(np.shape(db.df_windows[0]), np.shape(db.df_windows[1]))

    """
    Sliding window test

    data=[1,2,3,4,5,6,7,8,9,10]

    input_seq_length = 3

    output_seq_length = 2
    
    input, output = sliding_windows(data, input_seq_length, output_seq_length)

    idx= [0,1,2]
    for i in idx:
        print(input[i], output[i])

    """