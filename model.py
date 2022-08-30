# A3C Model

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Random weight initializer
def normalized_columns_initializer(weights, stdev=1.0):
    normalized_weights = torch.randn(weights.size())
    normalized_weights *= stdev / torch.sqrt(normalized_weights.pow(2).sum(1).expand_as(normalized_weights))  # Variance of the normalized columns is now stdev^2
    return normalized_weights


# Optimal weight initializer
def optimal_weights_initializer(model):
    classname = model.__class__.__name__
    
    # Convolution layer
    if classname.find('Conv') > -1:
        weight_shape = list(model.weight.data.size())
        
        # Determine weight boundaries
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        weight_bound = np.sqrt(6 / fan_in + fan_out)
        
        # Fill in weights and biases
        model.weight.data.uniform_(-weight_bound, weight_bound)
        model.bias.data.fill_(0)
    
    # Linear layer
    elif classname.find('Linear') > -1:
        weight_shape = list(model.weight.data.size())
        
        # Determine weight boundaries
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]        
        weight_bound = np.sqrt(6 / fan_in + fan_out)
        
        # Fill in weights and biases
        model.weight.data.uniform_(-weight_bound, weight_bound)
        model.bias.data.fill_(0)
        
# A3C Model
class ActorCriticModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCriticModel, self).__init__()
        # CRNN backbone
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm= nn.LSTMCell(32*3*3, 256)
        # Here we have an encoded state of size 256
        
        # Head of the critic
        self.critic = nn.Linear(256, 1)
        
        # Head of the actor
        self.actor = nn.Linear(256, num_actions)
        
        # Initialize zeights
        self.apply(optimal_weights_initializer)
        self.actor.weight.data = normalized_columns_initializer(self.actor.weight.data, 0.01) # Small variance for actor
        self.critic.weight.data = normalized_columns_initializer(self.actor.weight.data, 1) # High variance for critic
        self.actor.bias.data.fill_(0)
        self.critic.bias.data.fill_(0)        
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        # Put model in training mode
        self.train()
        
    def forward(self, inputs):
        state, (lstm_h, lstm_c) = inputs
        
        # Convolutions with ELU activations
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        
        # Flatten
        x = x.view(-1, 32*3*3)
        
        #LSTM
        (lstm_h, lstm_c)  = self.lstm(x, (lstm_h, lstm_c))
        x = lstm_h
        
        # Critic
        x_critic = self.critic(x)
        
        # Actor
        x_actor = self.actor(x)
        
        return x_critic, x_actor, (lstm_h, lstm_c)