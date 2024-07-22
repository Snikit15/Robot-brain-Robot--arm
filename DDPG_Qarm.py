import random 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs=15, num_actions=5, hidden_size=256, init_w=3e-4):
        super(PolicyNetwork, self).__init__()
        
        # Network layers
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        # Initialize weights
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # Use tanh to get values in [-1, 1] and scale them to [-0.1, 0.1]
        action = 0.1 * torch.tanh(x) #5 actions will be outputed
        return action

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs=15, num_actions=5, hidden_size=256, init_w=3e-4):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        if action.dim() == 2:  # If action is (batch_size, num_actions)
            action = action.unsqueeze(1)  # Add a singleton dimension
        x = torch.cat([state, action], dim=2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class OUNoise:
    def __init__(self, action_dimension, scale=0.05, mu=0, theta=0.15, sigma=0.275):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class ReplayBuffer:
    def __init__(self, capacity=2000000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, terminated, truncated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # Wrap reward, terminated, and truncated in lists
        self.buffer[self.position] = (state, action, reward, next_state, terminated, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated, truncated

    def __len__(self):
        return len(self.buffer)