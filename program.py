import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG.Agent import Agent
from DDPG.utils import *


def unpack_observation(obs):
    return obs['desired_goal'],\
            obs['achieved_goal'],\
            np.concatenate((obs['observation'], obs['desired_goal'])),\
            np.concatenate((obs['observation'], obs['achieved_goal']))



def prep_reward(a_goal, d_goal, success):
    d_goal = np.array(d_goal)
    a_goal = np.array(a_goal)
    distance = np.sqrt(np.sum((d_goal-a_goal)**2))
    if success:
        return 100

    if distance < 0.15:
        return 1 - 7 * distance

    return -1 * distance


env = NormalizedActionEnv(gym.make("FetchPush-v1"))

agent = Agent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
success_rates = []
success = 0
for episode in range(100):
    goal, achieved_goal, state, state_false = unpack_observation(env.reset())
    noise.reset()
    episode_reward = 0
    episode_success = 0

    for step in range(50):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_obs, reward, done, info = env.step(action)
        goal, achieved_goal, new_state, new_state_false = unpack_observation(new_obs)

        s = info['is_success']
        if s:
            success += 1
        reward = prep_reward(goal, achieved_goal, s)
        exp = (state, action, reward, new_state, done)
        agent.experience_replay.add_experience(exp)
        false_goal = achieved_goal.copy()
        false_reward = env.compute_reward(achieved_goal, false_goal, info)
        false_reward = prep_reward(false_goal, achieved_goal, True)

        false_exp = (state_false, action, false_reward, new_state_false, True)
        agent.experience_replay.add_experience(false_exp)

        if len(agent.experience_replay) > batch_size:
            agent.update(batch_size)

        state = new_state
        state_false = new_state_false
        episode_reward += reward
        # env.render()

        if s:
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    success_rates.append(success/(episode + 1))

    sys.stdout.write(
        "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                 np.mean(rewards[-10:])))


done = False
s = False

test_rewards = []
test_rates = []
test_successes = 0
number_of_tests = 100
for i in range(0, number_of_tests):
    obs = env.reset()
    goal, achieved_goal, state, state_false = unpack_observation(obs)
    test_reward = 0
    s = False
    done = False
    while not done and not s:
        action = agent.get_action(state)
        obs, reward, done, info = env.step(action)
        goal, achieved_goal, state, _ = unpack_observation(obs)
        s = info['is_success']
        if s:
            test_successes += 1
        reward = prep_reward(goal, achieved_goal, s)
        env.render()
        test_reward += reward

    test_rates.append(test_successes / (i + 1))
    test_rewards.append(test_reward)
    sys.stdout.write(
        "test: {}, reward: {}, average _reward: {} \n".format(i, np.round(test_reward, decimals=2),
                                                                 np.mean(test_rewards[-10:])))


print("-------------------------")
print('success:')

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


plt.plot(success_rates)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Success rate')
plt.show()

plt.plot(test_rates)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Test success rate')
plt.show()

plt.plot(test_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Test rewards')
plt.show()
