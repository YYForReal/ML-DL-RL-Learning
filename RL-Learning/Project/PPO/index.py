import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import os


# 定义神经网络模型
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        action_prob = torch.softmax(self.action_head(x), dim=-1)
        return action_prob


# 定义PPO算法
class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.002, gamma=0.99, epsilon=0.2):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(
            self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update_policy(self, states, actions, log_probs, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        advantages = torch.FloatTensor(advantages)

        new_action_probs = self.policy_network(states)
        new_dist = Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 +
                            self.epsilon) * advantages

        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 保存模型
    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print("save_model path", path)
        torch.save(self.policy_network.state_dict(), path)

    # 加载模型
    def load_model(self, path):
        if os.path.exists(path):
            print("load_model path", path)
            self.policy_network.load_state_dict(torch.load(path))
        else:
            print("load_model path", path, "not exists")
# 训练PPO Agent


def train_ppo_agent(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(state_size, action_size)
    # 加载模型
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, f'ppo_model_{env_name}.pth')
    # model_path = os.path.join(base_path, f'ppo_model.pth')

    if os.path.exists(model_path):
        agent.load_model(model_path)

    # agent.load_model('./ppo_model.pth')
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        # 计算优势值
        discounted_rewards = []
        advantage = 0
        for r in reversed(rewards):
            advantage = agent.gamma * advantage + r
            discounted_rewards.append(advantage)
        discounted_rewards.reverse()

        # 归一化优势值
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # 更新策略
        agent.update_policy(states, actions, log_probs, discounted_rewards)

        if episode % (num_episodes // 100) == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
    # 保存模型
    model_path = os.path.join(base_path, f'ppo_model_{env_name}.pth')

    agent.save_model(model_path)


# 测试PPO Agent并显示效果
def test_ppo_agent(env_name='CartPole-v1', num_episodes=10):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(state_size, action_size)
    # 加载模型
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, f'ppo_model_{env_name}.pth')
    if os.path.exists(model_path):
        agent.load_model(model_path)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()


# env_name = 'CartPole-v1'
env_name = 'MountainCar-v0'
# 训练PPO Agent
train_ppo_agent(env_name=env_name, num_episodes=1000)

# 测试PPO Agent并显示效果
test_ppo_agent(env_name=env_name, num_episodes=1)
