import random
from env import Action
import numpy as np
import copy


class Agent:
    def __init__(self, env):
        self.env = env

    def choose_action(self, state):
        # 在基类中，我们选择一个随机动作作为默认行为
        return random.choice([Action.STICK, Action.HIT])

    def train(self, num_episodes):
        # 在基类中，我们不执行任何训练，具体的算法将在子类中实现
        pass


# 蒙特卡洛代理
class MonteCarloAgent(Agent):
    def __init__(self, env, discount_factor=1, epsilon=1):
        super().__init__(env)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        dealer_space, player_space, _ = self.env.get_state_space()
        # N(s)是状态s被访问的次数
        # N(s, a)是状态s中动作a被选择的次数
        self.N = np.zeros((dealer_space, player_space, len(Action)))
        # G_s(s)是使用该状态获得的所有回报的总和
        self.G_s = np.zeros((dealer_space, player_space))
        # 状态的q值 == V
        self.q_values = np.zeros((dealer_space, player_space, len(Action)))
        # 设定epsilon初始值的辅助变量
        self.No = 100  # 初始值
    # 动作选择

    def dealerPolicy(self, s):
        """Lets act like the dealer."""
        if s.agent_sum >= 20:
            action = Action.STICK
        else:
            action = Action.HIT

        return action

    # 选择最佳动作

    def choose_best_action(self, state):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num
        q_values = self.q_values[dealer_sum][agent_sum]
        # print("q_values:", q_values)
        # print("acT np.argmax(q_values):", Action(np.argmax(q_values)))
        return Action(np.argmax(q_values))

    def choose_action(self, state):
        # epsilon-greedy policy
        # if np.random.rand() < self.epsilon:
        action = None
        if np.random.rand() < self.get_e(state):
            action = random.choice([Action.STICK, Action.HIT])
        else:
            action = self.choose_best_action(state)
        self.N[state.dealer_num, state.player_num, action.value] += 1
        return action

    # version1： 计算未来的累计回报，求平均

    def update_q_values_aver(self, episode):
        j = 0
        for state, action, _ in episode:
            dealer_sum = state.dealer_num
            agent_sum = state.player_num
            action_index = action.value

            # lncrement total return S(s) <- S(s)+ Gt
            self.N[dealer_sum][agent_sum][action_index] += 1
            Gt = sum([x[2] * (self.discount_factor ** i)
                     for i, x in enumerate(episode[j:])])
            self.G_s[dealer_sum][agent_sum] += Gt
            # lncrement counter N(s) <— N(s)＋1
            self.N[dealer_sum][agent_sum][action_index] += 1
            # Value is estimated by mean return V(s)= s(s)/ N(s)
            self.q_values[dealer_sum][agent_sum][action_index] = self.G_s[dealer_sum][agent_sum] / sum(
                self.N[dealer_sum][agent_sum, :])

            j += 1

    # version2： 计算未来的累计回报，增量式更新

    def update_q_values(self, episode):
        for t, (state, action, reward) in enumerate(episode):
            dealer_sum = state.dealer_num
            agent_sum = state.player_num
            action_index = action.value

            # 计算从当前时间步开始的未来累积回报
            Gt = sum([x[2] * (self.discount_factor ** i)
                     for i, x in enumerate(episode[t:])])

            # 计算增量
            delta = Gt - self.q_values[dealer_sum][agent_sum][action_index]

            # 更新 Q 值
            self.N[dealer_sum][agent_sum][action_index] += 1
            self.q_values[dealer_sum][agent_sum][action_index] += (
                1 / self.N[dealer_sum][agent_sum][action_index]) * delta

    def train(self, num_episodes, canshow=True):
        for episode_num in range(num_episodes):
            episode = []
            state, _, _ = self.env.reset()
            is_game_over = False
            while not is_game_over:
                # print("当前的state:", state)
                action = self.choose_action(state)
                next_state, reward, is_game_over = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            self.update_q_values(episode)

            # # 改进：动态调整ε
            if episode_num % int(num_episodes/10) == 0 and canshow:
                print(
                    f"Episode {episode_num}/{num_episodes} completed , now result_counts: {self.env.result_counts}")
                self.env.result_counts = [0] * 5

                # self.epsilon -= 0.1

        return self.q_values

    # 改进：动态调整ε
    def get_e(self, s):
        """et = N0/(N0 + N(st))"""
        # print("self.No:", self.No /
        #       ((self.No + sum(self.N[s.dealer_num, s.player_num, :]) * 1.0)))
        return self.No/((self.No + sum(self.N[s.dealer_num, s.player_num, :]) * 1.0))

    def get_value_function(self):
        return self.q_values


# Sarsa代理
class SARSAgent(Agent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env)
        dealer_space, player_space, _ = self.env.get_state_space()
        action_space = len(Action)
        self.q_values = np.zeros((dealer_space, player_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # episilon初始值设定
        self.No = 100  # 初始值

        # 记录次数
        self.iterations = 0
        self.wins = 0

    def choose_best_action(self, state):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num
        q_values = self.q_values[dealer_sum][agent_sum]
        return Action(np.argmax(q_values))

    def choose_action(self, state):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num

        if np.random.rand() < self.epsilon:
            action = random.choice([Action.STICK, Action.HIT])
        else:
            action = self.choose_best_action(state)

        return action

    def update_q_values(self, state, action, reward, next_state, next_action):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num
        action_index = action.value

        next_dealer_sum = next_state.dealer_num
        next_agent_sum = next_state.player_num
        next_action_index = next_action.value

        # 计算未来的累计回报 Q(S',A')
        next_q = self.q_values[next_dealer_sum][next_agent_sum][next_action_index]

        # 计算增量 R + γQ(S',A') - Q(S,A)
        delta = reward + self.gamma * next_q - \
            self.q_values[dealer_sum][agent_sum][action_index]

        # 更新 Q 值 Q(S,A) <- Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        self.q_values[dealer_sum][agent_sum][action_index] += self.alpha * delta

        return self.q_values

    def train(self, num_episodes, canshow=True):
        for e in range(num_episodes):
            state, _, _ = self.env.reset()
            action = self.choose_action(state)
            is_game_over = False
            next_action = action
            while not is_game_over:
                # Take action A, observe R, S'
                next_s, r, is_game_over = self.env.step(action)
                # 如果结束就没有了下一个状态了.
                if is_game_over:
                    q = self.q_values[state.dealer_num][state.player_num][action.value]
                    # 更新Q值
                    self.q_values[state.dealer_num][state.player_num][action.value] += self.alpha * (
                        r - q)
                    break
                # 下一个动作
                next_action = self.choose_action(next_s)
                # 更新Q值
                self.update_q_values(state, action, r, next_s, next_action)
                # 更新状态
                state = next_s
                action = next_action

            # update wins and iterations
            self.iterations += 1
            if r == 1:
                self.wins += 1

            # if e % 100000 == 0 and e != 0:
            #     print("Episode: %d, score: %f" %
            #           (e, (float(self.wins)/self.iterations)*100))
            if e % int(num_episodes/10) == 0 and canshow:
                self.epsilon = self.No / (e + self.No)
                print(
                    f"Episode {e}/{num_episodes} completed, epsilon:{self.epsilon} , now result_counts: {self.env.result_counts}")
                self.env.result_counts = [0] * 5

        return self.q_values

# Qlearning


class QLearningAgent(Agent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env)
        dealer_space, player_space, _ = self.env.get_state_space()
        action_space = len(Action)
        self.q_values = np.zeros((dealer_space, player_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # episilon初始值设定
        self.No = 100  # 初始值

        # 记录次数
        self.iterations = 0
        self.wins = 0

    def choose_best_action(self, state):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num
        q_values = self.q_values[dealer_sum][agent_sum]
        return Action(np.argmax(q_values))

    def choose_action(self, state):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num

        if np.random.rand() < self.epsilon:
            action = random.choice([Action.STICK, Action.HIT])
        else:
            action = self.choose_best_action(state)

        return action

    # q-learning的更新方式
    def update_q_values(self, state, action, reward, next_state, next_action):
        dealer_sum = state.dealer_num
        agent_sum = state.player_num
        action_index = action.value

        next_dealer_sum = next_state.dealer_num
        next_agent_sum = next_state.player_num

        # 计算未来的最大Q值 Q(S',A')
        max_q_next = np.max(self.q_values[next_dealer_sum][next_agent_sum])

        # 计算增量 R + γmaxQ(S',A') - Q(S,A)
        delta = reward + self.gamma * max_q_next - \
            self.q_values[dealer_sum][agent_sum][action_index]

        # 更新 Q 值 Q(S,A) <- Q(S,A) + α[R + γmaxQ(S',A') - Q(S,A)]
        self.q_values[dealer_sum][agent_sum][action_index] += self.alpha * delta

        return self.q_values

    def train(self, num_episodes, canshow=True):
        for e in range(num_episodes):
            state, _, _ = self.env.reset()
            action = self.choose_action(state)
            is_game_over = False
            next_action = action
            while not is_game_over:
                # Take action A, observe R, S'
                next_s, r, is_game_over = self.env.step(action)
                # 如果结束就没有了下一个状态了.
                if is_game_over:
                    q = self.q_values[state.dealer_num][state.player_num][action.value]
                    # 更新Q值
                    self.q_values[state.dealer_num][state.player_num][action.value] += self.alpha * (
                        r - q)
                    break
                # 下一个动作
                next_action = self.choose_action(next_s)
                # 更新Q值
                self.update_q_values(state, action, r, next_s, next_action)
                # 更新状态
                state = next_s
                action = next_action

            # update wins and iterations
            self.iterations += 1
            if r == 1:
                self.wins += 1

            # if e % 100000 == 0 and e != 0:
            #     print("Episode: %d, score: %f" %
            #           (e, (float(self.wins)/self.iterations)*100))
            if e % int(num_episodes/10) == 0 and canshow:
                self.epsilon = self.No / (e + self.No)
                print(
                    f"Episode {e}/{num_episodes} completed, epsilon:{self.epsilon} , now result_counts: {self.env.result_counts}")
                self.env.result_counts = [0] * 5

        return self.q_values
