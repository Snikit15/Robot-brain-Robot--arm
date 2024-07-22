import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import DDPG_Qarm
from DDPG_Qarm import PolicyNetwork, ValueNetwork, ReplayBuffer, OUNoise

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# init the network
state_dim = 15  # observation dimention
action_dim = 5  # action dimention
hidden_dim = 256  # hidden layer

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

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
#**********************weiter trainieren****************************
# #load the pretrained-model
# policy_net.load_state_dict(torch.load('policy_net.pth', map_location=device))
# value_net.load_state_dict(torch.load('value_net.pth', map_location=device))

# # change mode to train
# policy_net.train()
# value_net.train()
#**********************weiter trainieren****************************
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
#**********************weiter trainieren****************************
# with open('replay_buffer.pkl', 'rb') as f:
#     replay_buffer = pickle.load(f)
#**********************weiter trainieren****************************
ou_noise = OUNoise(action_dimension=action_dim)

MODULE_NAME = 'QArmEnv'
gym.envs.register(
    id='QArmEnv-v0',
    entry_point=f'{MODULE_NAME}:QArmEnv',
)

def convert_state_to_tensor(state):
    # Concatenate the values from the state dictionary
    # Ensure that the keys are in the same order as in the observation space definition
    state_values = np.concatenate([
        np.array(state[key]) for key in [
            'TCP', 'CUBE', 
            'TARGET_POSITION', 'STATE_JOINT_GRIPPER'
        ]
    ])
    
    # Convert to a PyTorch tensor
    return torch.FloatTensor(state_values).to(device)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_replay_buffer(buffer, path):
    with open(path, 'wb') as f:
        pickle.dump(buffer, f)

def train(env, episodes, max_steps_per_episode, batch_size=128):
    reward_list = [] 
    step_history = []  

    for episode in range(episodes):
        state = env.reset()[0]
        print("state[CUBE]",state["CUBE"])
        ou_noise.reset()
        episode_reward = 0
        step_count = 0

        for step in range(max_steps_per_episode):
            # get the state
            state_tensor = convert_state_to_tensor(state)
            state_tensor = state_tensor.unsqueeze(0)  
            action = policy_net(state_tensor)
            action = action.squeeze(0)  
            noise = torch.from_numpy(ou_noise.noise()).to(device)
            action = action + noise
            action = action.clamp(env.action_space.low[0], env.action_space.high[0])

            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())  # transform from GPU to CPU
            next_state_tensor = convert_state_to_tensor(next_state).view(1, -1)

            reward = torch.FloatTensor([reward])  
            replay_buffer.push(state_tensor, action.detach().cpu().numpy(), reward, next_state_tensor, terminated, truncated)  # save in CPU

            if len(replay_buffer) > batch_size:
                ddpg_update(batch_size)

            state = next_state
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        reward_list.append(episode_reward)
        step_history.append(step_count)
        print("Episode: {}, Total Reward: {}, Steps: {}".format(episode, episode_reward, step_count))

        plot_rewards(reward_list)
        plot_stepcount(step_history)
        save_model(policy_net, 'policy_net.pth')
        save_model(value_net, 'value_net.pth')
        save_replay_buffer(replay_buffer, 'replay_buffer.pkl')

    print("Training completed, models and replay buffer saved.")

if __name__ == "__main__":
    env = gym.make('QArmEnv-v0')
    train(env=env, episodes=5000, max_steps_per_episode=100, batch_size=128)