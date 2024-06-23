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
num_episodes = 100  # 迭代次数
epsilon = 1.0  # 初始探索率
epsilon_min = 0.2  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
learning_rate = 0.001  # 学习率
discount_factor = 0.9  # 折扣因子
batch_size = 64
memory_size = 10000
max_steps = 500
model_path = "./models/dqn_maze.pt"

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
        [1, 0, 1, 0, 0, 8, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
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
class ReplayBuffer:
    """
    经验回放缓存，用于存储和采样经验
    """
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

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
        self.memory = ReplayBuffer(memory_size)
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
        if np.random.rand() <= self.epsilon:
            valid_actions = []
            for action in range(self.action_size):
                next_state = update_state(state, action)
                if maze[next_state] != 1:  # 只选择不会撞墙的动作
                    valid_actions.append(action)
            if not valid_actions:
                return random.randrange(self.action_size)
            return random.choice(valid_actions)
        
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state).cpu().detach().numpy()
        
        # 将导致撞墙的动作值设置为负无穷大，以避免选择这些动作
        for action in range(self.action_size):
            next_state = update_state(state.cpu().numpy().astype(int).tolist(), action)
            if maze[next_state] == 1:
                act_values[action] = -float('inf')
        
        return np.argmax(act_values)

    def learn(self):
        # 检查经验回放缓存中的样本数量是否足够一个批次，如果不足则不进行学习
        if self.memory.size() < batch_size:
            return
        
        # 从经验回放缓存中随机采样一个批次的样本
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将采样的经验数据转换为张量，并移动到设备上（CPU或GPU）
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # 使用当前Q网络计算采样状态和动作的Q值
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 使用目标Q网络计算下一个状态的最大Q值
        next_q_values = self.target_model(next_states).max(1)[0]
        
        # 根据贝尔曼方程计算目标Q值
        target_q_values = rewards + (1 - dones) * discount_factor * next_q_values

        # 计算当前Q值和目标Q值之间的均方误差损失
        loss = F.mse_loss(q_values, target_q_values)

        # 反向传播，更新Q网络的参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 根据epsilon衰减策略，逐渐减少epsilon的值，减少探索，增加利用
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay


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
    if maze[state] == 8:
        return 1000
    elif maze[state] == 1:
        return -1
    else:
        return -0.1

# 计算与目标的距离
def distance_to_goal(state, goal):
    """
    计算当前状态与目标的曼哈顿距离
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

# 训练DQN模型
def train_dqn(agent, num_episodes, update_target_every=1000):
    """
    训练DQN智能体
    """
    step_count = 0  # 计步器
    all_rewards = []  # 存储所有episode的奖励
    all_losses = []   # 存储所有episode的损失
    for episode in tqdm(range(num_episodes), desc="Training"):
        maze = generate_single_map(maze_height, maze_width)
        state = (1, 1)
        state_flat = np.array(state).flatten()
        total_reward = 0
        for t in range(max_steps):
            # 动作选择
            action = agent.choose_action(state_flat, maze) 
            next_state = update_state(state, action) 
            reward = get_reward(next_state, maze)
            next_state_flat = np.array(next_state).flatten()
            done = (maze[next_state] == 8) # 判断是否走到了终点

            # 添加经验到回放缓存，并更新优先级
            agent.memory.add((state_flat, action, reward, next_state_flat, done)) 
            agent.learn() # 智能体根据经验学习

            # 更新状态，统计奖励
            state = next_state
            state_flat = next_state_flat
            total_reward += reward
            total_reward = round(total_reward, 2)
            step_count += 1
            
            # 定期更新目标网络
            if step_count % update_target_every == 0:
                agent.update_target_model()
            
            if done:
                break
        
        all_rewards.append(total_reward)
        loss = agent.learn()
        if loss is not None:
            all_losses.append(loss.item())
        
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
