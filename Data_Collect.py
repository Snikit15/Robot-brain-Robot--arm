import gymnasium as gym
import numpy as np
import pickle

def collect_data(env, episodes, max_steps_per_episode, action_clip_range=(-0.1, 0.1)):
    data = []

    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps_per_episode):
            # control_policy is a predefined control policy function
            action = control_policy(state)  # Replace here to use your control strategy
            # Ensure that the action is within the specified range
            action = np.clip(action, action_clip_range[0], action_clip_range[1])

            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Collecting the required data
            episode_data.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated
            })

            state = next_state

            if terminated or truncated:
                break
        
        data.append(episode_data)
    
    return data

def control_policy(state):
    # Here can be a deterministic control strategy based on some rules
    # Randomized actions are returned here as an example only
    return env.action_space.sample()

def save_data(data, filename='collected_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    env = gym.make('QArmEnv-v0')
    collected_data = collect_data(env, episodes=100, max_steps_per_episode=100, action_clip_range=(-0.1, 0.1))
    save_data(collected_data)

    print("Data collection completed, data saved.")
