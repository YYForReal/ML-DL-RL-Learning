from networks import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# 定义Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        print(state)
        # 若state为tuple类型，转为np array
        # if isinstance(state, tuple):
            # state = tuple(np.expand_dims(s, axis=0) if s.ndim == 0 else s for s in state)
            # state = np.concatenate(state)

        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # 计算TD误差
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values

        # 计算Actor和Critic的损失
        self.optimizer_actor.zero_grad()
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(td_errors.detach() * log_probs).mean()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss = td_errors.pow(2).mean()
        critic_loss.backward()
        self.optimizer_critic.step()
