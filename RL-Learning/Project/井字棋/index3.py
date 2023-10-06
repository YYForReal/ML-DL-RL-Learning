import random
import numpy as np

# 定义棋盘大小
BOARD_SIZE = 15
# 定义Q-learning学习率
LEARNING_RATE = 0.2
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

# 定义经验回放缓冲区
replay_buffer = []

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

class AI:
    def __init__(self):
        self.q_table = {}

    def choose_action(self, state):
        if random.random() < EXPLORATION_PROB:
            return random.choice(get_valid_actions(state))
        else:
            valid_actions = get_valid_actions(state)
            q_values = [self.q_table.get(str(state), {}).get(str(action), 0) for action in valid_actions]
            return valid_actions[np.argmax(q_values)]

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

# 打印棋盘
def print_board(board):
    for row in board:
        print(" ".join(["O" if cell == 1 else "X" if cell == -1 else "-" for cell in row]))
    print()

# 进行一次游戏
def play_game():
    board = Board()
    ai = AI()

    pre_state = None
    ai_action = None # 上次ai下棋的动作
    while True:
        # 玩家下棋
        player_action = input("请输入您要落子的位置，以逗号分隔（例如：1,2）：")
        try:
            player_action = tuple(map(int, player_action.split(',')))
            if player_action not in get_valid_actions(board.board):
                raise ValueError()
        except ValueError:
            print("输入有误，请重新输入！")
            continue

        board.board[player_action] = 1

        # 玩家下棋结束后的，就是这轮的
        if pre_state is not None:
            # pre_state是上次AI下棋前
            nxt_state = np.copy(board.board) # nxt_state是到玩家下棋结束，这次AI下棋前
            replay_buffer.append((pre_state, np.copy(ai_action), nxt_state))

        # 打印棋盘
        print_board(board.board)

        if board.is_game_over(player_action):
            # 若玩家胜利，回放整局过程并更新Q-table
            print("玩家获胜！")
            reward = -50  # 最后一步的惩罚最大
            for s, a, ns in reversed(replay_buffer):
                ai.update_q_table(s, a, ns, reward)
                reward += 1  # 逐步减小惩罚
            break


        # AI下棋
        pre_state = np.copy(board.board)
        ai_action = ai.choose_action(board.board)
        board.board[ai_action] = -1

        # 打印棋盘
        print_board(board.board)

        # 更新状态
        if board.is_game_over(ai_action):
            # 若AI胜利，回放整局过程并更新Q-table
            print("AI获胜！")
            reward = 50
            for s, a, ns in reversed(replay_buffer):
                ai.update_q_table(s, a, ns, reward)
                reward -= 1  # 逐步增大奖励
            break

        # 维持经验回放缓冲区大小
        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

# 开始与玩家对战
play_game()

print("================ 第2局 ===================")

play_game()
