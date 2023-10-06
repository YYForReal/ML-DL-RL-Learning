import random
import numpy as np
import os
import pickle


# 定义棋盘大小
BOARD_SIZE = 15
# 定义Q-learning学习率
LEARNING_RATE = 0.5
# 定义折扣因子
DISCOUNT_FACTOR = 0.9
# 定义探索概率
EXPLORATION_PROB = 0.2
# 定义经验回放缓冲区大小
REPLAY_BUFFER_SIZE = 1000

# 初始化棋盘
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Q-table表示状态-动作价值
q_table = {}


# 获取当前状态下可行的动作
def get_valid_actions(state):
    valid_actions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state[i][j] == 0:
                valid_actions.append((i, j))
    return valid_actions

# 定义Board类表示棋盘和游戏状态
class Board:
    def __init__(self):
        self.board = np.copy(board)

    def is_game_over(self, last_action):
        row, col = last_action
        color = self.board[row][col]
        directions = [(-1, 0), (0, -1), (-1, -1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or self.board[r][c] != color:
                    break
                count += 1
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or self.board[r][c] != color:
                    break
                count += 1
            if count >= 5:
                return True
        return False

# 定义AI类表示AI的行为
class AI:
    def __init__(self, role):
        self.role = role
        self.q_table = {}
        self.load_q_table()
        self.last_action = None
        self.last_state = None
        self.replay_buffer = []
        self.MAX_REPLAY_BUFFER_SIZE = 1000

    def choose_action(self, state, force_action = None):
        """
        AI根据当前的state进行决策，若传入force_action，则表示玩家插入，强制修改当前的action
        :param state: 棋盘
        :param force_action: 坐标
        :return: action
        """
        final_action = None
        if random.random() < EXPLORATION_PROB:
            final_action =  random.choice(get_valid_actions(state))
        else:
            valid_actions = get_valid_actions(state)
            q_values = [self.q_table.get(str(state), {}).get(str(action), 0) for action in valid_actions]
            final_action = valid_actions[np.argmax(q_values)]

        if force_action is not None :
            final_action = force_action

        if self.last_action is not None and self.last_state is not None:
            # 存储经验回放缓冲区
            self.replay_buffer.append((np.copy(self.last_state), np.copy(self.last_action), np.copy(state)))
            if len(self.replay_buffer) > self.MAX_REPLAY_BUFFER_SIZE:
                self.replay_buffer.pop(0)

        self.last_action = np.copy(final_action)
        self.last_state = np.copy(state)
        return final_action

    def update_q_table(self, state, action, reward, next_state):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = {}

        action_str = str(action)
        if action_str not in self.q_table[state_str]:
            self.q_table[state_str][action_str] = 0.0

        next_state_str = str(next_state)
        next_max_q = max(self.q_table.get(next_state_str, {}).values()) if next_state_str in self.q_table else 0.0
        updated_q = self.q_table[state_str][action_str] + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max_q - self.q_table[state_str][action_str])
        self.q_table[state_str][action_str] = updated_q

    def save_q_table(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(f"data/q_table_{self.role}.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(f"data/q_table_{self.role}.pkl"):
            with open(f"data/q_table_{self.role}.pkl", "rb") as f:
                self.q_table = pickle.load(f)


    def summary(self,max_reward,step):
        """
        经验回放总结
        :return:
        """
        reward = max_reward  # 最后一步的惩罚最大
        for s, a, ns in reversed(self.replay_buffer):
            self.update_q_table(s, a, ns, reward)
            reward += step  # 逐步减小惩罚

# 定义BlackAI类，继承自AI类
class BlackAI(AI):
    def __init__(self):
        super().__init__(role=1)  # 设置角色属性为1（黑色）


    # ... 可以根据需要覆盖父类的其他方法 ...

# 定义WhiteAI类，继承自AI类
class WhiteAI(AI):
    def __init__(self):
        super().__init__(role=-1)  # 设置角色属性为-1（白色）



# 打印棋盘
def print_board(board):
    for row in board:
        print(" ".join(["O" if cell == 1 else "X" if cell == -1 else "-" for cell in row]))
    print()

# 进行一次游戏
def play_game():
    board = Board()
    ai = WhiteAI()
    human_ai = BlackAI()

    pre_state = None
    ai_action = None # 上次ai下棋的动作


    print_board(board.board)
    while True:
        # 玩家输入
        player_action = input("请输入您要落子的位置，以逗号分隔（例如：1,2）：")
        try:
            player_action = tuple(map(int, player_action.split(',')))
            if player_action not in get_valid_actions(board.board):
                raise ValueError()
        except ValueError:
            print("输入有误，请重新输入！")
            continue

        # 强制：模拟human_ai下棋
        player_action = human_ai.choose_action(board.board,player_action)
        board.board[player_action] = 1

        # 打印棋盘
        print_board(board.board)

        if board.is_game_over(player_action):
            # 若玩家胜利，回放整局过程并更新Q-table
            print("玩家获胜！")
            ai.summary(-50,1)
            human_ai.summary(50,-1)
            break

        # AI下棋
        ai_action = ai.choose_action(board.board)
        board.board[ai_action] = -1

        # 打印棋盘
        print_board(board.board)

        # 更新状态
        if board.is_game_over(ai_action):
            # 若AI胜利，回放整局过程并更新Q-table
            print("AI获胜！")
            ai.summary(50,-1)
            human_ai.summary(-50,1)
            break

    # 游戏存档
    ai.save_q_table()
    human_ai.save_q_table()


# 进行一次游戏
def play_game2():
    board = Board()
    white_ai = WhiteAI()
    black_ai = BlackAI()

    pre_state = None
    ai_action = None # 上次ai下棋的动作


    print_board(board.board)
    while True:


        # 强制：模拟human_ai下棋
        black_ai_action = black_ai.choose_action(board.board)
        board.board[black_ai_action] = 1

        # 打印棋盘
        print_board(board.board)

        if board.is_game_over(black_ai_action):
            # 若玩家胜利，回放整局过程并更新Q-table
            print("black_ai 获胜！")
            black_ai.summary(-50,1)
            white_ai.summary(50,-1)
            break

        # 玩家输入
        player_action = input("请输入您要落子的位置，以逗号分隔（例如：1,2）：")
        try:
            player_action = tuple(map(int, player_action.split(',')))
            if player_action not in get_valid_actions(board.board):
                raise ValueError()
        except ValueError:
            print("输入有误，请重新输入！")
            continue

        # AI下棋
        white_ai_action = white_ai.choose_action(board.board,player_action)
        board.board[white_ai_action] = -1

        # 打印棋盘
        print_board(board.board)

        # 更新状态
        if board.is_game_over(white_ai_action):
            # 若AI胜利，回放整局过程并更新Q-table
            print("White （AI）获胜！")
            black_ai.summary(-50,1)
            white_ai.summary(50,-1)
            break

    # 游戏存档
    black_ai.save_q_table()
    white_ai.save_q_table()



print("================ 第1局 ===================")

# 开始与玩家对战
play_game()

print("================ 第2局 ===================")

play_game()

print("================ 第3局 ===================")

# 开始与玩家对战
play_game2()

print("================ 第4局 ===================")

play_game2()

