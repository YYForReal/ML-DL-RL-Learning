import torch
import torch.optim as optim
import math
import numpy as np
import random

class PER_DQN:
    def __init__(self, model, memory, cfg):

        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy策略相关参数
        self.sample_count = 0  # 用于epsilon的衰减计数
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update

        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        ## 复制参数到目标网络
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.memory = memory # SumTree 经验回放
        self.update_flag = False 
        
    def sample_action(self, state):
        ''' sample action with e-greedy policy
        '''
        self.sample_count += 1
        # epsilon 指数衰减
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # 根据Q值选择动作
        else:
            action = random.randrange(self.n_actions)
        return action

    def predict_action(self,state):
        ''' 预测动作
        '''
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item() 
        return action
    

    def update(self):
        if len(self.memory) < self.batch_size: # 不满足一个批量时，不更新策略
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # 采样一个batch
        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), idxs_batch, is_weights_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)

        # loss中根据优先度进行了加权
        loss = torch.mean(torch.pow((q_value_batch - expected_q_value_batch) * torch.from_numpy(is_weights_batch).to(self.device), 2))

        # loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)   

        abs_errors = np.sum(np.abs(q_value_batch.cpu().detach().numpy() - expected_q_value_batch.cpu().detach().numpy()), axis=1)
        # 需要更新样本的优先度
        self.memory.batch_update(idxs_batch, abs_errors) 

        # 反向传播
        self.optimizer.zero_grad()  
        loss.backward()
        # 梯度截断，防止梯度爆炸
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        if self.sample_count % self.target_update == 0: # 更新 target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())  
