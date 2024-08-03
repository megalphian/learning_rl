import gymnasium as gym
import numpy as np

from collections import deque
import torch

import matplotlib.pyplot as plt

from train_agent import Agent, train_agent

from enum import Enum

class sim_mode(Enum):
    TRAIN = 1
    TEST = 2

MODE = sim_mode.TEST

if MODE == sim_mode.TRAIN:

    ##### Uncomment to re-train the agent #####
    env = gym.make("LunarLander-v2")
    agent = Agent(state_size=8, action_size=4, seed=0)

    scores = train_agent(env, agent)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

elif MODE == sim_mode.TEST:
    
        ##### Uncomment to test the agent #####
        env = gym.make("LunarLander-v2", render_mode="human")
        agent = Agent(state_size=8, action_size=4, seed=0)
    
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint_lunar.pth'))
    
        for i in range(3):
            state,info = env.reset()
            for j in range(400):
                action = agent.act(state)
                env.render()
                state, reward,terminated,truncated,info= env.step(action)
                if terminated or truncated:
                    break 

env.close()


