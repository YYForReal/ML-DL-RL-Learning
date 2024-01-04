import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)


    def forward(self, state):
        return torch.softmax(self.fc2(torch.relu(self.fc1(state))), dim=-1)


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# 定义Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
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


# 创建环境和Agent
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 64
lr_actor = 0.001
lr_critic = 0.001
gamma = 0.99

agent = ActorCriticAgent(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma)

# 训练Agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 存储经验
        agent.update([state], [action], [reward], [next_state], [done])

        state = next_state

        if done:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            break

env.close()
