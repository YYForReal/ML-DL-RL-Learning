import numpy as np
import matplotlib.pyplot as plt

from maze_generation import maze_map

# 定义迷宫环境
# maze = np.array([
#     [0, 0, 0, 0, 0],
#     [0, -1, -1, 0, 0],
#     [0, 0, 0, -1, 0],
#     [0, -1, 0, -1, 0],
#     [0, 0, 0, 0, 0],
# ])

maze = np.array(maze_map)

# 定义Q-Learning算法的参数
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 20

# Q-Learning算法
Q = np.zeros((maze.shape[0], maze.shape[1], 4), dtype=float)  # 这里使用了4个动作，上、下、左、右
print(f"Q:{Q}")

def visualize_maze(maze, path=None):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.imshow(maze, cmap="Blues", interpolation="nearest")
    if path is not None:
        for i, j in path:
            ax.text(j, i, "X", ha='center', va='center', color='red', fontsize=15)
    plt.show()


def epsilon_greedy_policy(Q, state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(len(Q[state[0], state[1]]))
    else:
        return np.argmax(Q[state[0], state[1]])


all_rewards = []
for episode in range(num_episodes):
    state = (0, 0)  # 初始状态
    episode_rewards = 0
    print(episode)
    while state != (4, 4):  # 直到到达终点
        action = epsilon_greedy_policy(Q, state)
        y, x = state

        # 获取下一个状态
        if action == 0 and y > 0 and maze[y - 1, x] != -1:  # 上
            next_state = (y - 1, x)
        elif action == 1 and y < 4 and maze[y + 1, x] != -1:  # 下
            next_state = (y + 1, x)
        elif action == 2 and x > 0 and maze[y, x - 1] != -1:  # 左
            next_state = (y, x - 1)
        elif action == 3 and x < 4 and maze[y, x + 1] != -1:  # 右
            next_state = (y, x + 1)
        else:
            # 如果动作是走出边界或者撞墙，则不更新Q值，直接进入下一轮循环
            continue

        next_y, next_x = next_state

        # 更新Q值
        Q[y, x, action] = Q[y, x, action] + learning_rate * (
                    maze[next_state] + discount_factor * np.max(Q[next_y, next_x]) - Q[y, x, action])

        state = next_state
        episode_rewards += maze[state]

    all_rewards.append(episode_rewards)

# 训练完成后，可以使用训练好的Q值来找到最优路径
state = (0, 0)
optimal_path = [(0, 0)]

while state != (4, 4):
    action = np.argmax(Q[state[0], state[1]])
    y, x = state

    # 获取下一个状态
    if action == 0 and y > 0 and maze[y - 1, x] != -1:  # 上
        next_state = (y - 1, x)
    elif action == 1 and y < 4 and maze[y + 1, x] != -1:  # 下
        next_state = (y + 1, x)
    elif action == 2 and x > 0 and maze[y, x - 1] != -1:  # 左
        next_state = (y, x - 1)
    elif action == 3 and x < 4 and maze[y, x + 1] != -1:  # 右
        next_state = (y, x + 1)

    state = next_state
    optimal_path.append(next_state)

print("训练后的Q值：")
print(Q)

print("最优路径：")
print(optimal_path)

# 可视化迷宫和Q值的变化
visualize_maze(maze, optimal_path)

# 可视化奖励曲线
plt.plot(all_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.title('Training Progress')
plt.show()
