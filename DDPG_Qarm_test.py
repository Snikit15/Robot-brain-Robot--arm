import gymnasium as gym
import abc
import glob
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from types import LambdaType
from collections import deque
from collections import namedtuple
import random 
import time
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, ByteTensor
Tensor = FloatTensor
import pickle



print(torch.cuda.is_available())
device = torch.device("cpu")

def plot_rewards(reward_list, filename='rewards_plot.png'):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    plt.savefig(filename)

def plot_stepcount(step_history, filename='steps_plot.png'):
    plt.figure(2)
    plt.clf()
    plt.title('Steps every episode')
    plt.xlabel('Episode')
    plt.ylabel('Step Count')
    plt.plot(step_history)
    plt.savefig(filename) 


# init the network
state_dim = 21  # observation dimention
action_dim = 5  # action dimention
hidden_dim = 256  # hidden layer


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs=21, num_actions=5, hidden_size=256, init_w=3e-4):
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
        action = 0.1 * torch.tanh(x)
        return action

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs=21, num_actions=5, hidden_size=256, init_w=3e-4):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        #print(state.shape)
        if action.dim() == 2:  # If action is (batch_size, num_actions)
            action = action.unsqueeze(1)  # Add a singleton dimension
        #print(action.shape)
        x = torch.cat([state, action], dim=2)
        #print(x.shape)
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
    
def ddpg_update(batch_size, gamma=0.99, min_value=-np.inf, max_value=np.inf, soft_tau=1e-2):
    state, action, reward, next_state, terminated, truncated = replay_buffer.sample(batch_size)

    # make sure all the info in device
    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    terminated = torch.FloatTensor(terminated).to(device)
    truncated = torch.FloatTensor(truncated).to(device)
    
    # Calculate the 'done' flag
    done = terminated + truncated
    done = torch.clamp(done, max=1.0)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    #print(target_value.size())
    expected_value = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * gamma * target_value
    #expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
    
    # print("value size:", value.size())
    # print("expected_value size:", expected_value.size())

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


policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)

target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)

# syncronize the target network with network
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

# optimizer
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
value_criterion = nn.MSELoss()

replay_buffer = ReplayBuffer()

ou_noise = OUNoise(action_dimension=action_dim)
MODULE_NAME = 'QArmEnv_fetch_test'
gym.envs.register(
    id='QArmEnv-v0',
    entry_point=f'{MODULE_NAME}:QArmEnv',
)

# Initialize env
env = gym.make('QArmEnv-v0')

def convert_state_to_tensor(state):
    # Concatenate the values from the state dictionary
    # Ensure that the keys are in the same order as in the observation space definition
    state_values = np.concatenate([
        np.array(state[key]) for key in [
            'TCP', 'TARGET', 'RELATIVE', 
            'CUBE_target_position', 'RELATIVE_cube_target_position', 'GRIPPER'
        ]
    ])
    
    # Convert to a PyTorch tensor
    return torch.FloatTensor(state_values).to(device)
    #lse:
        # Handle the case when state is already a tensor
        #return state.to(device)  # Assuming state is already on the correct device

def test_model(env, policy_net, episodes, max_steps_per_episode):
    reward_list = []  # save return every epoch
    step_history = []  # save steps every epoch

    for episode in range(episodes):
        state = env.reset()[0]  # reset the environment
        episode_reward = 0
        step_count = 0

        for step in range(max_steps_per_episode):
            state_tensor = convert_state_to_tensor(state)  # save the state as tensor
            state_tensor = state_tensor.unsqueeze(0)  # add dimension
            action = policy_net(state_tensor)  # generate action with policy net
            action = action.squeeze(0).detach().numpy()  # delete the dimension and turn to numpy array
            next_state, reward, terminated, truncated, _ = env.step(action)  # do the action and get the state and the reward next step
            state = next_state
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        reward_list.append(episode_reward)
        step_history.append(step_count)
        print(f"Test Episode: {episode}, Total Reward: {episode_reward}, Steps: {step_count}")

    # plot the result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_list)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(step_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.tight_layout()
    plt.show()

# load the policy network
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net.load_state_dict(torch.load('policy_net.pth', map_location=device))
policy_net.eval()  # set to evaluation mode

# test the mode
test_model(env, policy_net, episodes=10, max_steps_per_episode=100)

