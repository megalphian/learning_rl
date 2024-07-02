import gymnasium as gym
import numpy as np

import train_agent

##### Uncomment to re-train the agent #####
# env = gym.make("FrozenLake-v1", desc=None, is_slippery=False, map_name="4x4")
# train_agent.train_agent(env)

env = gym.make("FrozenLake-v1", desc=None, is_slippery=False, map_name="4x4", render_mode="human")
q_table = np.load('q_table.npy')
state, info = env.reset()
reward = 0

for step in range(100):
    env.render()
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()


