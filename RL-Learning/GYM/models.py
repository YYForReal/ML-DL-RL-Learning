from networks import Actor, Critic, ReplayBuffer,DQN,PrioritizedBuffer
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
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    def select_action(self, state):
        raise NotImplementedError
    
    def load_model(self, path):
        raise NotImplementedError
    
    def save_model(self, path):
        raise NotImplementedError

# 定义Actor-Critic Agent
class ActorCriticAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, 
                 n_updates_critic=1, m_updates_actor=1):
        super(ActorCriticAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.n_updates_critic = n_updates_critic #每隔n步更新一次critic
        self.m_updates_actor = m_updates_actor # 每隔m步更新一次actor
        self.update_times = 0 # 记录更新次数

    def select_action(self, state, eps=0.20):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        # print("action_probs",action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

        # eps-greedy
        # if np.random.randn() < eps:
        #     return np.random.choice(range(action_probs.shape[1])), dist.log_prob(action)
        # else:    
        #     # print("action",action.item())

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        self.update_times += 1

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values

        if self.update_times % self.n_updates_critic == 0:
            self.optimizer_critic.zero_grad()
            critic_loss = td_errors.pow(2).mean()
            critic_loss.backward()
            self.optimizer_critic.step()

        if  self.update_times % self.m_updates_actor == 0:
            self.optimizer_actor.zero_grad()
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            actor_loss = -(td_errors.detach() * log_probs).mean()
            actor_loss.backward()
            self.optimizer_actor.step()

    def load_model(self, path, epoch=0):
        print("load Model from ",path)
        if os.path.exists(os.path.join(path, f'actor-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth')):
            self.actor.load_state_dict(torch.load(os.path.join(path, f'actor-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth')))
        if os.path.exists(os.path.join(path, f'critic-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth')):
            self.critic.load_state_dict(torch.load(os.path.join(path, f'critic-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth')))

    def save_model(self, path, epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), os.path.join(path, f'actor-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, f'critic-{self.n_updates_critic}-{self.m_updates_actor}-{epoch}.pth'))
   

class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=1000000,target_update = 5):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0] 
        action_dim = env.action_space.n

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        # 当前网络
        self.model = DQN(state_dim, action_dim).to(self.device)
        # 目标网络
        self.target_model = DQN(state_dim, action_dim).to(self.device) 
        self.target_update = target_update # 每隔target_update步更新目标网络

        # 将目标网络初始化为当前网络
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.sample_count = 0 # 采样步数

        
    def select_action(self, state, eps=0.20,mask_action_space=None):
        self.sample_count += 1

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        # 判定随机
        if (np.random.randn() < eps):
            if mask_action_space is not None:
                # 从mask里面取一个 mask_action_space = np.array( [2,3,4,6,7])
                action = np.random.choice(mask_action_space)
                return action,qvals            
            return self.env.action_space.sample(),qvals
        
        return action,qvals
        # 判定有无mask
        # if mask_action_space is not None:
        #     # print("before ",qvals)
        #     # TODO 依据qval的价值，从mask [2,3,4,5]里面取一个
        #     # 将非mask中的动作的价值设为负无穷大，这样它们就不会被选择
        #     qvals[:, ~np.isin(range(len(qvals[0])), mask_action_space)] = float('-inf')
        #     # print("after ",qvals)
        #     # input("----")
        #     # 根据q价值从mask中选择动作
        #     # print("self.env.action_space.n",self.env.action_space.n)
        #     action_probs = torch.softmax(qvals, dim=1).cpu().detach().numpy()
        #     # print("action_probs",action_probs)
        #     action = np.random.choice(self.env.action_space.n, p=action_probs.flatten())
        #     # print("action",action)
        #     return action,qvals
        # return action,qvals

    def compute_loss(self, batch):     
        # states, actions, rewards, next_states, dones = batch
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device)

        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)



        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1).to(self.device)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        print("Curr_Q",curr_Q.shape)
        print("expected_Q",expected_Q.shape)
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    # def update(self, batch_size):
    #     # print("start update",batch_size)
    #     batch = self.replay_buffer.sample(batch_size)
    #     loss = self.compute_loss(batch)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     # target network update
    #     for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
    #         target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    #     return loss.item()

    def update(self, batch_size, share_agent=None):
        # 从经验回放中采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        # 转换成张量（便于GPU计算）
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) 
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) 
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) 
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) 
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) 
        # 计算 Q 的实际值
        q_value_batch = self.model(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        # 计算 Q 的估计值，即 r+\gamma Q_max
        next_max_q_value_batch = self.target_model(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        # 计算损失
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch.detach())  
        # 梯度清零，避免在下一次反向传播时重复累加梯度而出现错误。
        self.optimizer.zero_grad()  
        # 反向传播
        loss.backward()
        # clip避免梯度爆炸
        for param in self.model.parameters():  
            param.grad.data.clamp_(-1, 1)
        # 更新优化器
        self.optimizer.step() 
        # 每C(target_update)步更新目标网络
        if self.sample_count % self.target_update == 0: 
            self.target_model.load_state_dict(self.model.state_dict())   

        return loss.item()


    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

