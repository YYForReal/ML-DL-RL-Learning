import random

def generate_maze(n):
    maze = [[1 for _ in range(n)] for _ in range(n)]  # 创建一个n x n的迷宫地图，并初始化为墙

    # 将起点和终点标记为通路
    maze[0][0] = 0
    maze[n-1][n-1] = 0

    wall_count = n * n - 2

    # DFS函数用于生成迷宫地图
    def dfs(x, y):
        nonlocal wall_count
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] == 1:
                # 随机决定是否打通该墙
                if random.random() < 0.8 and wall_count > n*n/3 :  # 70%的概率打通该墙，且墙的数量不少于n//2
                    maze[nx][ny] = 0
                    wall_count -= 1
                    dfs(nx, ny)

    # 从起点开始生成迷宫
    dfs(0, 0)

    return maze

# 设置迷宫的大小
n = 15

# 生成迷宫地图
maze_map = generate_maze(n)

# 打印生成的迷宫地图
for row in maze_map:
    print(row)


