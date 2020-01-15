import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG.Agent import Agent
from DDPG.utils import *

# env = NormalizedEnv(gym.make("FetchPush-v1"))
env = gym.make("FetchPush-v1")

agent = Agent(env)
noise = OUNoise(env.action_space)
batch_size = 150
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        exp = (state, action, reward, new_state, done)
        agent.experience_replay.add_experience(exp)

        if len(agent.experience_replay) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward
        env.render()
        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
