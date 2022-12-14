import torch 
import torch.nn as nn 
import torch.nn.functional as F

""" Since the data is almost perioding a simple neural network is likely to do the trick. Hence we experiment with 2 neural network designs
1. Multi Layer Perception; 'FeedForwardModel' which is simply made up of 3 linear layers
2. LSTM model; 'LSTMModel' which is a 2 layer model, where first layer is an lstm layer followed by a linear layer for producing an output

LSTM models though are high performing are also notoriously slow to train, and is likely to be overkill for this dataset. Since the dataset is well curated, 
it is likely the Multi Layer Perceptron simple model is going to outperform LSTM (when trained for the same number of epochs).
"""

class FeedForwardModel(nn.Module):
    def __init__(self, args):
        super(FeedForwardModel, self).__init__()
        self.args = args

        self.input_size = self.args.input_sequence_length
        self.output_size = self.args.output_sequence_length
        self.hidden_size = self.args.hidden_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x= self.relu1(self.layer1(x))
        x= self.relu2(self.layer2(x))
        out = self.layer3(x)
        return out


class LSTMModel(nn.Module):

    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.args = args

        self.output_size = self.args.output_sequence_length
        self.num_layers = self.args.num_layers
        self.input_size = self.args.input_feature_size
        self.hidden_size = self.args.hidden_size
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
    
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    