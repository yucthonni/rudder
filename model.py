import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM
import torch.optim as optim

class value_pred(nn.Module):
    def __init__(self,args):
        super(value_pred, self).__init__()
        self.input_dim = args.state_dim + args.action_dim
        self.output_dim = 1
        self.hidden_dim =args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        self.dvc = torch.device('cpu')
        if args.cuda:
            self.dvc = torch.device('cuda')
        
        self.lstm = LSTM(self.input_dim,self.hidden_dim,self.num_layers,).to(self.dvc)
        
        self.linear = nn.Linear(self.hidden_dim,self.output_dim).to(self.dvc)
    def forward(self, obs, act):
        lstm_out, _ = self.lstm(torch.cat([obs,act],dim=-1))
        out = self.linear(lstm_out)
        return out
    