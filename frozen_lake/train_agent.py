import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

ALPHA = 0.8
GAMMA = 0.9

min_epsilon = 0.01
max_epsilon = 1.0
decay = 0.005

def compute_next_q_value(old_q, reward, next_state_q):
    return old_q + ALPHA * (reward + GAMMA * next_state_q - old_q)

def epsilon_greedy_policy(env, q_table, state, epsilon):
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(q_table[state])
    else:
        return env.action_space.sample()
        
    
def reduce_epsilon(epoch):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)

def train_agent(env):

    action_space = env.action_space.n
    state_space = env.observation_space.n
    q_table = np.zeros([state_space, action_space])
    epoch_plot_tracker = []
    total_reward_plot_tracker = []

    env.reset(seed=42)
    rewards = []
    log_interval = 100

    epochs = 1000
    epsilon = max_epsilon

    for episode in range(epochs):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(env, q_table, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            old_q = q_table[state, action]
            next_state_q = np.max(q_table[new_state])

            q_table[state, action] = compute_next_q_value(old_q, reward, next_state_q)
            total_reward += reward
            state = new_state
        
        epsilon = reduce_epsilon(episode)
        rewards.append(total_reward)
        epoch_plot_tracker.append(episode)
        total_reward_plot_tracker.append(np.sum(rewards))

        if episode % log_interval == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")
        
    env.close()
    np.save("q_table.npy", q_table)
    plt.plot(epoch_plot_tracker, total_reward_plot_tracker)
    plt.xlabel("Epochs")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Epochs")
    plt.show()