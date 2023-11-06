import copy


class GridWorld:
    """ 网格世界，坐标系原点(0,0) 
    """

    def __init__(self, ncol=4, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        self.target_grids = [(0, 0), (ncol-1, nrow-1)]  # 存储目标区域
        # 状态数量
        self.state_nums = ncol * nrow
        # 4种动作, actions[0]:上,actions[1]:下, actions[2]:左, actions[3]:右。
        self.actions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 状态转移概率矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        # state 是一维表示的
        self.P = self.init_P()

    def init_P(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.state_nums)]
        # 定义在左上角
        for i in range(self.nrow):
            for j in range(self.ncol):
                # 遍历四个方向
                for a in range(4):
                    # 采取行动后进入的状态s'、奖励r都是固定只有一种
                    psa = 1
                    # 位置在目标状态,因为不需要继续交互,任何动作奖励都为0
                    if (i, j) in self.target_grids:
                        P[i * self.ncol +
                            j][a] = [(psa, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置(边界越界后依旧保持不动)
                    next_x = min(self.ncol - 1, max(0, j + self.actions[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + self.actions[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在目标状态，Done
                    if (next_x, next_y) in self.target_grids:
                        done = True
                    P[i * self.ncol + j][a] = [(psa, next_state, reward, done)]
        return P


class CliffWalkingEnv:
    """ 悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
