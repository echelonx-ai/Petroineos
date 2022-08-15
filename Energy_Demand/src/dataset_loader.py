import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class EnergyDataset(Dataset):
    def __init__(self, args, dataframe, split): #, target, features, sequence_length=5):
        self.args = args
        self.dataframe = dataframe
        self.split = split
        self.input_sequence_length = self.args.input_sequence_length
        self.output_sequence_length = self.args.output_sequence_length

        self.scaler = MinMaxScaler()
        # splitting data into 80-20 i.e. 70% training, 10% val, 20% testing
        if self.split=='train':
            self.df
        elif self.split=='val':
            self.df 
        else:
            self.df

        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

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

    data=[1,2,3,4,5,6,7,8,9,10]

    input_seq_length = 3

    output_seq_length = 2
    
    input, output = sliding_windows(data, input_seq_length, output_seq_length)