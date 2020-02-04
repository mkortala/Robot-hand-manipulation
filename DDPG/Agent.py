import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import Actor, Critic
from DDPG.utils import *
from torch.autograd.variable import Variable


class Agent:
    def __init__(self, env, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, max_memory=int(1e6)):
        obs = env.reset()
        self.num_states = obs['desired_goal'].shape[0] + obs['observation'].shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.action_max = env.action_space.high[0]

        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1)

        self.target_actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.target_critic = Critic(self.num_states + self.num_actions, hidden_size, 1)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.experience_replay = ExperienceReplay(max_memory)
        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

    def update(self, size):
        states, actions, rewards, next_states, _ = self.experience_replay.sample(size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        with torch.no_grad():
            next_actions = self.target_actor.forward(next_states)
            q_next = self.target_critic.forward(next_states, next_actions).detach()
            target_q = rewards.reshape((128,1)) + self.gamma * q_next
            target_q = target_q.detach()
            c = 1/ (1-self.gamma)
            target_q = torch.clamp(target_q, -c, 0)

        real_q = self.critic.forward(states, actions)
        dif = (target_q - real_q)
        critic_loss = dif.pow(2).mean()
        real_actions = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, real_actions).mean()
        actor_loss += (real_actions/self.action_max).pow(2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



        # update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
