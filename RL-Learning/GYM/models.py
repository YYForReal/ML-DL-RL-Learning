from networks import Actor, Critic, BasicBuffer,ConvDQN,DQN
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os

# 定义一个通用的Agent类，具有动作选择、加载保存初始模型的功能，供其他agent继承
class BaseAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_action(self, state):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError

# 定义Actor-Critic Agent
class ActorCriticAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma):
        super(ActorCriticAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        # state 转 np.ndarray
        state = np.array(state)
        state = torch.from_numpy(list(state)).float().unsqueeze(0)
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

    def load_model(self, path,epoch=0):
        # 兼容性
        if os.path.exists(os.path.join(path, 'actor.pth')):
            self.actor.load_state_dict(torch.load(os.path.join(path, f'actor-{epoch}.pth')))
        if os.path.exists(os.path.join(path, 'critic.pth')):
            self.critic.load_state_dict(torch.load(os.path.join(path, f'critic-{epoch}.pth')))

    def save_model(self, path,epoch=0):
        torch.save(self.actor.state_dict(), os.path.join(path, f'actor-{epoch}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, f'critic-{epoch}.pth'))
        

class DQNAgent:

    def __init__(self, env, use_conv=False, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0] 
        action_dim = env.action_space.n

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            # self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            # self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.model = DQN(state_dim, action_dim).to(self.device)
            self.target_model = DQN(state_dim, action_dim).to(self.device)
        
        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        
    def select_action(self, state, eps=0.20,mask_action_space=None):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if (np.random.randn() < eps):
            if mask_action_space is not None:
                # 从mask里面取一个
                action = np.random.choice(mask_action_space)
                return action,qvals
            return self.env.action_space.sample(),qvals
        return action,qvals

    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1).to(self.device)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    def update(self, batch_size):
        # print("start update",batch_size)
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return loss.item()

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

