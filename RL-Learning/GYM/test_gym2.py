import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from models import ActorCriticAgent


# 创建环境和Agent
# env = gym.make('CartPole-v1')
env = gym.make("ALE/DonkeyKong-v5",render_mode="rgb_array")

state_dim = env.observation_space.shape
print(state_dim ) # (210, 160, 3)
# 兼容地转成1维数据
state_dim = state_dim[0] * state_dim[1] * state_dim[2]

action_dim = env.action_space.n
hidden_dim = 64
lr_actor = 0.001
lr_critic = 0.001
gamma = 0.99
seed = 12345




agent = ActorCriticAgent(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma)
# 训练Agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset(seed=seed)
    total_reward = 0

    while True:
        # 转化state
        # ValueError: expected sequence of length 210 at dim 1 (got 4)        
        # state = torch.tensor(state, dtype=torch.float32)

        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # 转化state
        next_state = np.array(next_state).flatten()
        # 存储经验
        agent.update([state], [action], [reward], [next_state], [done])

        state = next_state

        if done:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            break

env.close()
