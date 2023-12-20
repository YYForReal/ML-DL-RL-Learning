import random
from enum import Enum


random.seed(1)


class Color(Enum):
    RED = 0
    BLACK = 1


class Action(Enum):
    STICK = 0
    HIT = 1


class Card(object):
    value = 0
    color = Color.BLACK


class State(object):
    player_num = 0
    dealer_num = 0
    is_player_turn = True


class Envi:
    def __init__(self):
        self.player_cards = []  # 玩家牌
        self.dealer_cards = []  # 庄家牌
        self.is_player_turn = True
        self.is_game_over = False
        self.reward = 0
        self.result_counts_description = [
            "玩家爆牌", "庄家爆牌", "玩家>庄家", "庄家>玩家", "平局"]
        self.result_counts = [0]*5  # 记录游戏结果

    def reset(self):
        self.player_cards = []
        self.dealer_cards = []
        self.is_player_turn = True
        self.is_game_over = False
        self.reward = 0
        self.start_game()
        return self.get_state(), 0, False

    def start_game(self):
        # 开局发黑牌
        self.player_cards.append(self.take_card(Color.BLACK))
        self.dealer_cards.append(self.take_card(Color.BLACK))
        self.player_cards.append(self.take_card(Color.BLACK))
        self.dealer_cards.append(self.take_card(Color.BLACK))

    def take_card(self, color=None):
        deck = [{'value': value} for value in range(1, 11)]  # 1-10之间
        card = random.choice(deck)
        if color is None:
            color = Color.RED if random.random() < 0.4 else Color.BLACK
        card['color'] = color
        return card

    def step(self, action, state=None):
        # print("action:", action)
        # print("player's cards:", self.player_cards)
        # print("dealer's cards:", self.dealer_cards)
        # print("is player's turn:", self.is_player_turn)
        if self.is_game_over:
            return self.get_state(), self.reward, True

        if action == Action.HIT:
            self.player_hit()
        elif action == Action.STICK:
            self.player_stick()

        return self.get_state(), self.reward, self.is_game_over

    def player_hit(self):
        # print("player_hit")
        if not self.is_player_turn:
            return
        card = self.take_card()  # 玩家取牌
        self.player_cards.append(card)
        self.update()  # 更新状态和奖励

    def player_stick(self):
        self.is_player_turn = False  # 玩家弃牌
        self.dealer_play()  # 庄家操作纳入环境中
        self.update()  # 更新状态和奖励

    def dealer_play(self):
        # 庄家策略：只要小于20点，一直取牌
        while self.get_cards_sum(self.dealer_cards) < 20:
            card = self.take_card()
            self.dealer_cards.append(card)

    def get_cards_sum(self, cards):
        cards_sum = sum(card['value']
                        for card in cards if card['color'] == Color.BLACK)
        return cards_sum

    def get_state(self):
        state = State()
        state.player_num = self.get_cards_sum(self.player_cards)
        state.dealer_num = self.dealer_cards[0]['value']  # 只能看庄家的第一张牌
        state.is_player_turn = self.is_player_turn
        return state

    def update(self):
        player_sum = self.get_cards_sum(self.player_cards)
        dealer_sum = self.get_cards_sum(self.dealer_cards)
        # 玩家爆牌
        if player_sum > 21 or player_sum < 1:
            self.reward = -1
            self.is_game_over = True
            self.result_counts[0] += 1
            # print("玩家爆牌")
        # 庄家爆牌
        elif not self.is_player_turn and dealer_sum > 21:
            self.reward = 1
            self.is_game_over = True
            self.result_counts[1] += 1
            # print("庄家爆牌")
        # 玩家>庄家
        elif not self.is_player_turn and player_sum > dealer_sum:
            self.reward = 1
            self.is_game_over = True
            self.result_counts[2] += 1
            # print("玩家胜利")
        # 庄家>玩家
        elif not self.is_player_turn and player_sum < dealer_sum:
            self.reward = -1
            self.is_game_over = True
            self.result_counts[3] += 1
            # print("庄家胜利")
        # 平局
        elif not self.is_player_turn and player_sum == dealer_sum:
            self.reward = 0
            self.is_game_over = True
            self.result_counts[4] += 1
            # print("平局")
        # 继续游戏
        else:
            self.reward = 0

    def get_state_space(self):
        return (11, 22, 2)  # 庄家点数11种，玩家点数22种，玩家回合2种

# 使用示例：
# game = Envi()
# state, reward, done = game.reset()
# print("Player's cards:", game.player_cards)
# print("Dealer's cards:", game.dealer_cards)
# print("State:", state.player_num, state.dealer_num, state.is_player_turn)
# print("Reward:", reward)
# print("Is game over?", done)

# state, reward, done = game.step(Action.HIT)
# print("Player's cards after hitting:", game.player_cards)
# print("State:", state.player_num, state.dealer_num, state.is_player_turn)
# print("Reward:", reward)
# print("Is game over?", done)

# state, reward, done = game.step(Action.STICK)
# print("Dealer's cards after playing:", game.dealer_cards)
# print("State:", state.player_num, state.dealer_num, state.is_player_turn)
# print("Reward:", reward)
# print("Is game over?", done)
