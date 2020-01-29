from abc import ABC

import numpy as np
import gym
from collections import deque
import random


class ExperienceReplay:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add_experience(self, exp):
        self.buffer.append(exp)

    def sample(self, sample_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        exp_sample = random.sample(self.buffer, sample_size)

        for exp in exp_sample:
            state, action, reward, next_state, done = exp
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period = 100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def transform_state(self):
        s = self.state
        ds = self.theta * (self.mu - s) + self.sigma * np.random.randn(self.action_dim)
        self.state = s + ds
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.transform_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)

        return np.clip(action + ou_state, self.low, self.high)


class NormalizedEnv(gym.ActionWrapper):

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2.
        act_b = (self.action_space.high + self.action_space.low)/2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/2.
        return act_k_inv * (action - act_b)


class RewardWrap(gym.RewardWrapper):

    def reward(self, reward):
        if reward >= 0:
            return 4*reward

        return reward
