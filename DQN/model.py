import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np

class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, fc_factor = 5):
        
        """Initialize a QNetwork object.

        Params
        ======
            state_size (int): dimension of the agent state
            action_size (int): dimension of each action
            fc_factor (int): dimension of hidden layer
        """
        
        super(QNetwork, self).__init__()
        inter_num_nodes = state_size * fc_factor
        self.linear_1 = nn.Linear(state_size, inter_num_nodes)
        self.af_1 = F.relu
        self.linear_2 = nn.Linear(inter_num_nodes, inter_num_nodes)
        self.af_2 = F.relu
        self.linear_3 = nn.Linear(inter_num_nodes, action_size)
    
    def forward(self, state):
        
        """ Do one forward propagation
        
        Params
        ======
            state (numpy.array): current state of the agent
        """
        
        x = self.af_1(self.linear_1(state))
        x = self.af_2(self.linear_2(x))
        action = self.linear_3(x)
        return action
