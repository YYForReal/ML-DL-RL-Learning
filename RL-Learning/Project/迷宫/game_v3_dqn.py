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

# Hyperparameters
num_episodes = 500  # 迭代次数
epsilon = 1.0  # 初始探索率
epsilon_min = 0.3  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
learning_rate = 0.001  # 学习率
discount_factor = 0.9  # 折扣因子
batch_size = 64
memory_size = 10000
max_steps = 1000
model_path = "./models/dqn_maze.pt"

# Maze setup
def generate_random_maze(height, width):
    maze = np.zeros((height, width))
    maze[1:-1, 1:-1] = np.random.choice([0, 1], size=(height-2, width-2), p=[0.55, 0.45])
    # 四边都是墙
    maze[0, :] = 1  # top wall
    maze[-1, :] = 1  # bottom wall
    maze[:, 0] = 1  # left wall
    maze[:, -1] = 1  # right wall

    # 起始点和终点起码是空地
    maze[1, 1] = 0  # start point
    maze[height-2, width-2] = 8  # end point

    return maze

maze_height, maze_width = 10, 10

# DQN Network
class DQN(nn.Module):
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

# Memory Replay Buffer
class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)

# AI Agent
class Agent:
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
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

    def choose_action(self, state, maze):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state).cpu().detach().numpy()

        # Check for valid actions
        for action in range(self.action_size):
            next_state = update_state(state, action)
            if maze[next_state] == 1:  # 如果下一步是墙，将该动作的概率视为0
                act_values[action] = -float('inf')

        return np.argmax(act_values)

    def learn(self):
        if self.memory.size() < batch_size:
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * discount_factor * next_q_values

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

def update_state(state, action):
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

def get_reward(state, maze):
    if maze[state] == 8:
        return 10
    elif maze[state] == 1:
        return -1
    else:
        return -0.1

def distance_to_goal(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def train_dqn(agent, num_episodes):
    for episode in range(num_episodes):
        maze = generate_random_maze(maze_height, maze_width)
        state = (1, 1)
        state_flat = np.array(state).flatten()
        total_reward = 0
        for t in range(max_steps):
            action = agent.choose_action(state_flat, maze)
            next_state = update_state(state, action)
            reward = get_reward(next_state, maze)
            if maze[next_state] == 1:  # 如果走向墙壁，返回原位置并扣分
                next_state = state
                reward = -1
            next_state_flat = np.array(next_state).flatten()
            done = (maze[next_state] == 8)
            agent.memory.add((state_flat, action, reward, next_state_flat, done))
            agent.learn()
            state = next_state
            state_flat = next_state_flat
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")
                break
        else:  # 如果走到了最大步数，计算距离惩罚
            total_reward -= distance_to_goal(state, (maze_height-2, maze_width-2))
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward} (Timeout)")

    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save_model(model_path)
    print("Model saved to", model_path)

def draw_maze(maze):
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

def draw_agent(agent_state):
    row, col = agent_state
    pygame.draw.circle(screen, (255, 0, 0), (col * cell_size + cell_size //
                       2, row * cell_size + cell_size // 2), cell_size // 3)

def draw_agent_console(agent_state, maze):
    row, col = agent_state
    maze_with_agent = maze.copy()
    maze_with_agent[row, col] = 2  # 用数字2表示智能体在迷宫中的位置
    for row in range(maze_height):
        for col in range(maze_width):
            char = "#" if maze_with_agent[row,
                                          col] == 1 else "A" if maze_with_agent[row, col] == 2 else " "
            print(char, end=" ")
        print()

def show_all(agent_state, maze):
    # 展示pygame的
    screen.fill((0, 0, 0))
    draw_maze(maze)
    draw_agent(agent_state)
    pygame.display.flip()
    # 展示控制台的
    clear_console()
    draw_agent_console(agent_state, maze)

def clear_console():
    if os.name == 'nt':  # Windows系统
        os.system('cls')
    else:  # 其他系统（如Linux、macOS等）
        os.system('clear')

def run_maze(agent, maze):
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

    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(2, 4)

    # Check if model exists and load it
    if not args.retrain and os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded from", model_path)
    else:
        # Train the agent
        if os.path.exists(model_path) and args.retrain:
            agent.load_model(model_path)
            print("Model loaded from", model_path)
        print("Training model...")
        train_dqn(agent, num_episodes)

    # Pygame setup
    pygame.init()
    cell_size = 40
    screen = pygame.display.set_mode(
        (maze_width * cell_size, maze_height * cell_size))
    pygame.display.set_caption("Maze")

    # Test trained agent with Pygame visualization
    maze = generate_random_maze(maze_height, maze_width)
    run_maze(agent, maze)

    # Quit Pygame
    pygame.quit()
