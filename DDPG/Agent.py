import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import Actor, Critic
from DDPG.utils import *
from torch.autograd.variable import Variable


class Agent:
    def __init__(self, env, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2, max_memory=50000):
        obs = env.reset()
        self.num_states = obs['desired_goal'].shape[0] + obs['observation'].shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)

        self.target_actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.target_critic = Critic(self.num_states + self.num_actions, hidden_size, 1)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.experience_replay = ExperienceReplay(max_memory)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        obs = state['observation']
        goal = state['desired_goal']
        state = np.concatenate((obs, goal))
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self, size):
        states, actions, rewards, next_states, _ = self.experience_replay.sample(size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        vals = self.critic.forward(states, actions)
        next_actions = self.target_actor.forward(next_states)
        next_Q = self.target_critic.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(vals, Qprime)

        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
