import os
import numpy as np
import random
import pygame
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import argparse
from tqdm import tqdm

# 超参数设置
num_episodes = 500  # 迭代次数
epsilon = 1.0  # 初始探索率
epsilon_min = 0.1  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
learning_rate = 0.001  # 学习率
discount_factor = 0.9  # 折扣因子
batch_size = 64
memory_size = 10000
max_steps = 500
model_path = "./models/dqn_maze.pt"
update_target_every = 1000  # 更新目标网络的频率

maze_height, maze_width = 10, 10

# 固定迷宫生成函数
def generate_single_map(height, width):
    """
    生成一个固定的迷宫地图
    """
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 8, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    return maze

# DQN网络结构
class DQN(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 经验回放缓存
class PrioritizedReplayBuffer:
    """
    优先经验回放缓存，用于存储和采样经验
    """
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        self.epsilon = 1e-6

    def add(self, experience, priority):
        self.memory.append(experience)
        self.priorities.append(priority + self.epsilon)

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def size(self):
        return len(self.memory)

# 强化学习智能体
class Agent:
    """
    DQN智能体
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.epsilon = epsilon
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """
        更新目标网络的参数
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        """
        保存模型参数
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        加载模型参数
        """
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

    def choose_action(self, state, maze):
        """
        选择动作，避免撞墙
        """
        # 随机选择动作
        if np.random.rand() <= self.epsilon:
            valid_actions = []
            for action in range(self.action_size):
                next_state = update_state(state, action)
                if maze[next_state] != 1:  # 只选择不会撞墙的动作
                    valid_actions.append(action)
            if not valid_actions:
                return random.randrange(self.action_size)
            return random.choice(valid_actions)
        
        # 预测动作值
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state).cpu().detach().numpy()
        
        # 将导致撞墙的动作值设置为负无穷大，以避免选择这些动作
        for action in range(self.action_size):
            next_state = update_state(state.cpu().numpy().astype(int).tolist(), action)
            if maze[next_state] == 1:
                act_values[action] = -float('inf')
        
        return np.argmax(act_values)

    def learn(self):
        """
        学习更新Q网络
        """
        if self.memory.size() < batch_size:
            return
        batch, indices = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatFloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatFloatTensor(np.array(dones)).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * discount_factor * next_q_values

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

        priorities = (target_q_values - q_values).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices, priorities)

        return loss

 
# 更新状态
def update_state(state, action):
    """
    根据动作更新状态
    """
    row, col = state
    if action == 0:  # 上
        row = max(row - 1, 0)
    elif action == 1:  # 下
        row = min(row + 1, maze_height - 1)
    elif action == 2:  # 左
        col = max(col - 1, 0)
    elif action == 3:  # 右
        col = min(col + 1, maze_width - 1)
    return (row, col)

# 获取奖励
def get_reward(state, maze):
    """
    根据状态获取奖励
    """
    goal = (maze_height - 2, maze_width - 2)
    if maze[state] == 8:
        return 100
    elif maze[state] == 1:
        return -1
    else:
        # 使用曼哈顿距离来计算奖励，鼓励智能体靠近目标
        distance = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        return -0.1 + (1.0 / (distance + 1))

# 训练DQN模型
def train_dqn(agent, num_episodes, update_target_every=1000):
    step_count = 0  # 计步器，用于记录总步数
    all_rewards = []  # 存储所有episode的总奖励
    all_losses = []   # 存储所有episode的损失
    for episode in tqdm(range(num_episodes), desc="Training"):  # 使用tqdm显示训练进度
        maze = generate_single_map(maze_height, maze_width)  # 生成一个新的迷宫
        state = (1, 1)  # 初始化智能体的位置
        state_flat = np.array(state).flatten()  # 将状态展平为一维数组
        total_reward = 0  # 初始化总奖励
        for t in range(max_steps):  # 限制每个episode的最大步数
            action = agent.choose_action(state_flat, maze)  # 选择动作
            next_state = update_state(state, action)  # 更新状态
            reward = get_reward(next_state, maze)  # 获取奖励
            if maze[next_state] == 1:  # 如果走向墙壁，返回原位置并扣分
                next_state = state
                reward = -1
            next_state_flat = np.array(next_state).flatten()  # 将下一状态展平为一维数组
            done = (maze[next_state] == 8)  # 检查是否到达终点

            # 添加经验到回放缓存，并更新优先级
            agent.memory.add((state_flat, action, reward, next_state_flat, done), 100 + reward)  
            loss = agent.learn()  # 学习更新Q网络
            state = next_state  # 更新当前状态
            state_flat = next_state_flat  # 更新当前状态的一维表示
            total_reward += reward  # 累加奖励
            total_reward = round(total_reward, 2)  # 保留两位小数
            step_count += 1  # 增加步数计数
            
            # 定期更新目标网络
            if step_count % update_target_every == 0:
                agent.update_target_model()
            
            if done:  # 如果到达终点，结束当前episode
                break
        
        all_rewards.append(total_reward)  # 存储当前episode的总奖励
        if loss is not None:
            all_losses.append(loss.item())  # 存储当前episode的损失
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")
    
    return all_rewards, all_losses

# 绘制迷宫
def draw_maze(maze):
    """
    绘制迷宫
    """
    for row in range(maze_height):
        for col in range(maze_width):
            if maze[row, col] == 0:
                color = (255, 255, 255)  # 白色
            elif maze[row, col] == 1:
                color = (0, 0, 0)  # 黑色
            elif maze[row, col] == 8:
                color = (0, 255, 0)  # 绿色
            pygame.draw.rect(screen, color, (col * cell_size,
                             row * cell_size, cell_size, cell_size))

# 绘制智能体
def draw_agent(agent_state):
    """
    绘制智能体
    """
    row, col = agent_state
    pygame.draw.circle(screen, (255, 0, 0), (col * cell_size + cell_size //
                       2, row * cell_size + cell_size // 2), cell_size // 3)

# 控制台绘制智能体
def draw_agent_console(agent_state, maze):
    """
    在控制台绘制智能体
    """
    row, col = agent_state
    maze_with_agent = maze.copy()
    maze_with_agent[row, col] = 2  # 用数字2表示智能体在迷宫中的位置
    for row in range(maze_height):
        for col in range(maze_width):
            char = "#" if maze_with_agent[row,
                                          col] == 1 else "A" if maze_with_agent[row, col] == 2 else " "
            print(char, end=" ")
        print()

# 清空控制台
def clear_console():
    """
    清空控制台
    """
    if os.name == 'nt':  # Windows系统
        os.system('cls')
    else:  # 其他系统（如Linux、macOS等）
        os.system('clear')

# 显示所有
def show_all(agent_state, maze):
    """
    显示迷宫和智能体
    """
    screen.fill((0, 0, 0))
    draw_maze(maze)
    draw_agent(agent_state)
    pygame.display.flip()
    clear_console()
    draw_agent_console(agent_state, maze)

# 运行迷宫
def run_maze(agent, maze):
    """
    运行迷宫，测试训练好的智能体
    """
    state = (1, 1)
    state_flat = np.array(state).flatten()
    for _ in range(max_steps):
        if maze[state] == 8:
            break
        action = agent.choose_action(state_flat, maze)
        next_state = update_state(state, action)
        if maze[next_state] == 1:  # 如果走向墙壁，返回原位置并扣分
            next_state = state
        state = next_state
        state_flat = np.array(state).flatten()
        show_all(state, maze)
        time.sleep(0.1)
    show_all(state, maze)
    time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", default=False, help="retrain the model from scratch")
    args = parser.parse_args()

    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(2, 4)

    # 检查是否存在模型并加载
    if not args.retrain and os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded from", model_path)
    else:
        # 训练智能体
        if os.path.exists(model_path) and args.retrain:
            agent.load_model(model_path)
            print("Model loaded from", model_path)
        all_rewards, all_losses = train_dqn(agent, num_episodes)

    # Pygame设置
    pygame.init()
    cell_size = 40
    screen = pygame.display.set_mode(
        (maze_width * cell_size, maze_height * cell_size))
    pygame.display.set_caption("Maze")

    # 测试训练好的智能体
    maze = generate_single_map(maze_height, maze_width)
    run_maze(agent, maze)

    # 退出Pygame
    pygame.quit()
