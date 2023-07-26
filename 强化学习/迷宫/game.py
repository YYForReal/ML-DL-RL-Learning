import numpy as np
import random
import pygame
import time

# 迷宫地图，0表示可通行，1表示障碍物
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

# 迷宫尺寸
maze_height, maze_width = maze.shape

# AI智能体
class Agent:
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions

    def choose_action(self, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(Q_table[self.state[0], self.state[1], :])

    def update_state(self, new_state):
        self.state = new_state

# 创建迷宫界面
def draw_maze():
    for row in range(maze_height):
        for col in range(maze_width):
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

# 在迷宫中绘制AI智能体
def draw_agent(agent):
    row, col = agent.state
    pygame.draw.circle(screen, (255, 0, 0), (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2), cell_size // 3)

# 更新AI智能体在迷宫中的位置
def update_agent(agent, action):
    row, col = agent.state
    if action == 0:  # 上
        row = max(row - 1, 0)
    elif action == 1:  # 下
        row = min(row + 1, maze_height - 1)
    elif action == 2:  # 左
        col = max(col - 1, 0)
    else:  # 右
        col = min(col + 1, maze_width - 1)
    new_state = (row, col)
    return new_state

# 不同的数字表示的方位
def getChinesefromNum(action):
    res = ""
    if action == 0:  # 上
        res = "上"
    elif action == 1:  # 下
        res = "下"
    elif action == 2:  # 左
        res = "左"
    else:  # 右
        res = "右"
    return res


# 运行AI智能体在迷宫中的最优路径
def run_maze(agent):
    agent.state = (0, 0)  # 初始化智能体的状态为起始点
    screen.fill((0, 0, 0))
    pygame.time.delay(500)

    while agent.state != (maze_height - 1, maze_width - 1):  # 智能体到达目标点结束
        action = np.argmax(Q_table[agent.state[0], agent.state[1], :])  # 根据Q值表选择最优动作
        new_state = update_agent(agent, action)
        draw_maze()
        draw_agent(agent)
        pygame.display.flip()
        time.sleep(0.5)
        agent.update_state(new_state)
    draw_maze()
    draw_agent(agent)
    time.sleep(0.5)


# 初始化Q值表
Q_table = np.zeros((maze_height, maze_width, 4))


# Q-Learning算法
def q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor):
    global visualize
    for episode in range(num_episodes):
        agent.state = (0, 0)  # 初始化智能体的状态为起始点
        score = 0
        steps = 0
        path = []
        while agent.state != (maze_height - 1, maze_width - 1):  # 智能体到达目标点结束
            action = agent.choose_action(epsilon)
            new_state = update_agent(agent, action)
            path.append(getChinesefromNum(action))
            reward = -1 if maze[new_state] == 0 else -5  # 根据新状态更新奖励
            Q_table[agent.state[0], agent.state[1], action] += learning_rate * (reward + discount_factor * np.max(Q_table[new_state]) - Q_table[agent.state[0], agent.state[1], action])
            agent.update_state(new_state)
            score += reward
            steps += 1

        # 输出当前的episode和最佳路径长度
        best_path_length = int(-score / 5)
        if episode%10 == 0 :
          print(f"Episode: {episode}, Path Length: {steps}")
          print(f"path:{path}")

# Pygame初始化
pygame.init()
cell_size = 40
screen = pygame.display.set_mode((maze_width * cell_size, maze_height * cell_size))
pygame.display.set_caption("Maze")

# 定义智能体
agent = Agent((0, 0), [0, 1, 2, 3])

# 运行Q-Learning算法
num_episodes = 200
epsilon = 0.1
learning_rate = 0.1
discount_factor = 0.9
q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor)

run_maze(agent)

# 关闭Pygame
pygame.quit()
