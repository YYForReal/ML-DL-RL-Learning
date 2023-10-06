import numpy as np
import random
import pygame
import time
import os

"""
如果能够在自己电脑跑，还可以通过pygame看到窗口效果！
最近开始学DQN啦，希望有同窗大佬能带我一起讨论。
个人主页: https://blog.csdn.net/HYY_2000
"""
num_episodes = 2000  # 迭代次数
epsilon = 0.1  # 随机率
learning_rate = 0.1  # 学习率
discount_factor = 0.9

# 5 * 5
# maze = np.array([
#     [1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1],
# ])
# 10 * 10
# maze = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
#     [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
#     [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])


# 清空控制台输出
def clear_console():
    if os.name == 'nt':  # Windows系统
        os.system('cls')
    else:  # 其他系统（如Linux、macOS等）
        os.system('clear')


# 迷宫尺寸
maze_height, maze_width = maze.shape

init_position = {"x": 1, "y": 1}
end_position = {"x": maze_height - 2, "y": maze_width-2}

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
            pygame.draw.rect(screen, color, (col * cell_size,
                             row * cell_size, cell_size, cell_size))

# 在迷宫中绘制AI智能体
def draw_agent(agent):
    row, col = agent.state
    pygame.draw.circle(screen, (255, 0, 0), (col * cell_size + cell_size //
                       2, row * cell_size + cell_size // 2), cell_size // 3)

# 控制台的绘制
def draw_agent_console(agent):
    row, col = agent.state
    maze_with_agent = maze.copy()
    maze_with_agent[row, col] = 2  # 用数字2表示智能体在迷宫中的位置
    for row in range(maze_height):
        for col in range(maze_width):
            char = "#" if maze_with_agent[row, col] == 1 else "A" if maze_with_agent[row, col] == 2 else " "
            print(char, end=" ")
        print()
        
def show_all():
    # 展示pygame的
    draw_maze()
    draw_agent(agent)
    # 展示控制台的
    clear_console()
    draw_agent_console(agent)


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
    action_dict = {0: "上", 1: "下", 2: "左", 3: "右"}
    return action_dict.get(action, "")

# 运行AI智能体在迷宫中的最优路径
def run_maze(agent):
    agent.state = (init_position["x"], init_position["y"])  # 初始化智能体的状态为起始点
    screen.fill((0, 0, 0))
    pygame.time.delay(500)

    while agent.state != (end_position["x"], end_position["y"]):  # 智能体到达目标点结束
        action = np.argmax(
            Q_table[agent.state[0], agent.state[1], :])  # 根据Q值表选择最优动作
        new_state = update_agent(agent, action)
        show_all()
        pygame.display.flip()
        time.sleep(0.5)
        agent.update_state(new_state)
    # 结束了之后最后画一次
    show_all()
    time.sleep(0.5)


# 初始化Q值表
Q_table = np.zeros((maze_height, maze_width, 4))

# Q-Learning算法
def q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor):
    global visualize
    for episode in range(num_episodes):
        agent.state = (init_position["x"], init_position["y"])  # 初始化智能体的状态为起始点
        score = 0
        steps = 0
        path = []
        while agent.state != (end_position["x"], end_position["y"]):  # 智能体到达目标点结束
            action = agent.choose_action(epsilon)
            new_state = update_agent(agent, action)

            path.append(getChinesefromNum(action))

            # 如果设置成-5，那么相比撞墙，他不会选择绕路绕5格以上的路（惩罚5以上）。
            # reward = -1 if maze[new_state] == 0 else -5  # 根据新状态更新奖励

            reward = -1 if maze[new_state] == 0 else -100  # 根据新状态更新奖励

            # 陷入局部最优
            # distance_to_goal = abs(new_state[0] - (maze_height - 1)) + abs(new_state[1] - (maze_width - 1))
            # reward = -distance_to_goal if maze[new_state] == 0 else -999999999999999  # 根据新状态更新奖励

            # reward = (0 - distance_to_goal / (maze_height + maze_width)) if maze[new_state] == 0 else -999  # 根据新状态更新奖励

            Q_table[agent.state[0], agent.state[1], action] += learning_rate * \
                (reward + discount_factor *
                 np.max(Q_table[new_state]) - Q_table[agent.state[0], agent.state[1], action])
            agent.update_state(new_state)
            score += reward
            steps += 1

        # 输出当前的episode和最佳路径长度
        best_path_length = int(-score / 5)
        if episode % 10 == 0:
            print(f"重复次数: {episode}, 路径长度: {steps}")
            print(f"移动路径: {path}")


# Pygame初始化
pygame.init()
cell_size = 40
screen = pygame.display.set_mode(
    (maze_width * cell_size, maze_height * cell_size))
pygame.display.set_caption("Maze")

# 定义智能体
agent = Agent((1, 1), [0, 1, 2, 3])

# 运行Q-Learning算法
q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor)

run_maze(agent)

# 关闭Pygame
pygame.quit()

