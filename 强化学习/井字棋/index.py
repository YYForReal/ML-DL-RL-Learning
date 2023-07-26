import numpy as np
import random


class FourInARowGame:
    def __init__(self):
        self.board_size = 4
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.players = [1, -1]  # Player 1 and Player 2
        self.current_player = 1
        self.game_over = False

    def get_valid_moves(self):
        return [col for col in range(self.board_size) if self.board[0, col] == 0]

    def get_valid_moves(self):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0:
                    valid_moves.append((x, y))
        return valid_moves

    def is_board_full(self):
        return np.all(self.board != 0)

    def make_move(self, action):
        if not self.game_over:
            x, y = action
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0:
                self.board[x, y] = self.current_player

                winner = self.check_winner()
                if winner is not None or self.is_board_full():
                    self.game_over = True
                else:
                    self.current_player = -self.current_player

    def check_winner(self):
        for player in self.players:
            # Check rows
            for row in range(self.board_size):
                for col in range(self.board_size - 3):
                    if all(self.board[row, col + i] == player for i in range(4)):
                        return player

            # Check columns
            for col in range(self.board_size):
                for row in range(self.board_size - 3):
                    if all(self.board[row + i, col] == player for i in range(4)):
                        return player

            # Check diagonals (top-left to bottom-right)
            for row in range(self.board_size - 3):
                for col in range(self.board_size - 3):
                    if all(self.board[row + i, col + i] == player for i in range(4)):
                        return player

            # Check diagonals (bottom-left to top-right)
            for row in range(3, self.board_size):
                for col in range(self.board_size - 3):
                    if all(self.board[row - i, col + i] == player for i in range(4)):
                        return player

        if np.all(self.board != 0):
            return 0  # Game is a draw

        return None

    def is_game_over(self):
        return self.game_over

    def get_state(self):
        return self.board.copy()

    def get_reward(self, player):
        winner = self.check_winner()
        if winner is None:
            return 0
        elif winner == player:
            return 1
        else:
            return -1

    def get_current_player(self):
        return self.current_player

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = random.choice(self.players)
        self.game_over = False

    def show(self):
        print("Current Board:")
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 1:
                    print(" X ", end="")
                elif self.board[row, col] == -1:
                    print(" O ", end="")
                else:
                    print(" . ", end="")
            print()
        print()


class QAgent:
    def __init__(self, board_size, player=0, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self.board_size = board_size
        self.player = player
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}

    def get_q_value(self, state, action):
        state_str = str(state.reshape(-1))
        if state_str not in self.q_table:
            self.q_table[state_str] = {}

        action_str = str(action)
        if action_str not in self.q_table[state_str]:
            self.q_table[state_str][action_str] = 0.0

        return self.q_table[state_str][action_str]

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            # 随机选择一个动作
            return random.choice(valid_moves) if valid_moves else random.choice(self.get_all_actions())
        else:
            q_values = [self.get_q_value(state, action)
                        for action in valid_moves]
            if not q_values:
                return random.choice(self.get_all_actions())  # 随机选择一个动作
            max_q = max(q_values)
            best_actions = [action for action, q_value in zip(
                valid_moves, q_values) if q_value == max_q]
            return random.choice(best_actions)

    def get_all_actions(self):
        return [(x, y) for x in range(self.board_size) for y in range(self.board_size)]

    def update_q_table(self, state, action, reward, next_state):
        state_str = str(state.reshape(-1))
        if state_str not in self.q_table:
            self.q_table[state_str] = {}

        action_str = str(action)
        if action_str not in self.q_table[state_str]:
            self.q_table[state_str][action_str] = 0.0

        next_max_q = max(self.q_table[str(next_state.reshape(-1))].values()) if str(
            next_state.reshape(-1)) in self.q_table else 0.0
        updated_q = self.q_table[state_str][action_str] + self.learning_rate * (
            reward + self.discount_factor * next_max_q - self.q_table[state_str][action_str])
        self.q_table[state_str][action_str] = updated_q


class BlackAgent(QAgent):
    def __init__(self, board_size, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        super().__init__(board_size, player=1, epsilon=epsilon,
                         learning_rate=learning_rate, discount_factor=discount_factor)


class WhiteAgent(QAgent):
    def __init__(self, board_size, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        super().__init__(board_size, player=-1, epsilon=epsilon,
                         learning_rate=learning_rate, discount_factor=discount_factor)


def test(agent):
    print("============================ Testing =================================")
    game.reset()
    print(f"agent's player: {agent.player}")

    while not game.is_game_over():
        if game.current_player == agent.player:
            # Agent's turn
            state = game.get_state()
            valid_moves = game.get_valid_moves()
            action = agent.choose_action(state, valid_moves)
            print(f"Agent's move: {action}")
            game.make_move(action)
        else:
            # User's turn
            while True:
                try:
                    x = int(input("Enter the row (0 to 3): "))
                    y = int(input("Enter the column (0 to 3): "))
                    action = (x, y)
                    if action in game.get_valid_moves():
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Try again.")

            game.make_move(action)

        game.show()  # 展示当前对战情况

    winner = game.check_winner()
    if winner == 0:
        print("It's a draw!")
    else:
        print(f"Player {winner} wins!")



if __name__ == "__main__":
    # 初始化两个Agent，分别对应黑子和白子
    black_agent = BlackAgent(board_size=4)
    white_agent = WhiteAgent(board_size=4)

    num_episodes = 1000
    game = FourInARowGame()
    agent = QAgent(board_size=game.board_size)
    # 训练
    num_episodes = 10000
    for episode in range(num_episodes):
        game.reset()
        print("========================================")
        print(f"训练次数：{episode}")
        while not game.is_game_over():
            if game.current_player == black_agent.player:
                state = game.get_state()
                valid_moves = game.get_valid_moves()
                action = black_agent.choose_action(state, valid_moves)
                game.make_move(action)
            else:
                state = game.get_state()
                valid_moves = game.get_valid_moves()
                action = white_agent.choose_action(state, valid_moves)
                game.make_move(action)


        # 更新Q值
        winner = game.check_winner()
        if winner == black_agent.player:
            reward = 1.0
        elif winner == white_agent.player:
            reward = -1.0
        else:
            reward = 0.0

        black_agent.update_q_table(state, action, reward, game.get_state())
        white_agent.update_q_table(state, action, -reward, game.get_state())

    test(white_agent)

    
# Training


# for episode in range(num_episodes):
#     game.reset()
#     print("========================================")
#     print(f"训练次数：{episode}")
#     agent.current_player = random.choice(game.players)
#     while not game.is_game_over():
#         state = game.get_state()
#         valid_moves = game.get_valid_moves()
#         action = agent.choose_action(state, valid_moves)
#         game.make_move(action)
#         # game.show()
#         next_state = game.get_state()
#         reward = game.get_reward(agent.current_player)
#         agent.update_q_table(state, action, reward, next_state)

# Testing


