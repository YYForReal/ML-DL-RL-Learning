import copy
import time


class PolicyIteration:
    """ 策略迭代算法 """

    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子
        self.init_policy()

    def init_policy(self):
        state_nums = self.env.state_nums
        # 初始化价值为0
        self.v = [0] * state_nums
        # 初始化为均匀随机策略
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(state_nums)]

    # 策略评估
    def policy_evaluation(self, show=False):
        cnt = 1  # 计算迭代次数
        # start_time = time.perf_counter()   # 计算收敛时间
        state_nums = self.env.state_nums
        while 1:
            max_diff = 0
            # 新的状态价值函数 数组
            new_v = [0] * state_nums
            for s in range(state_nums):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                # 尝试这个状态的所有动作
                for a in range(4):
                    qsa = 0.0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        # 应用贝尔曼方程
                        qsa += p * (r + self.gamma * self.v[next_state])
                    qsa_list.append(self.pi[s][a] * qsa)
                # 状态价值函数和动作价值函数之间的关系
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            # print(f"自举：{new_v}")
            self.v = new_v  # 自举

            if max_diff < self.theta:
                break  # 满足收敛条件,退出评估迭代
            cnt += 1
        # 计算收敛时间
        # end_time = time.perf_counter()
        # convergence_time = (end_time - start_time) * 1000
        # print(f"策略评估进行 {cnt} 轮后完成，收敛时间：{convergence_time} 毫秒")
        if show:
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    # 为了输出美观,保持输出6个字符
                    print('%6.6s' %
                          ('%.3f' % self.v[i * self.env.ncol + j]), end=' ')
                print()
            print()

    # 策略提升
    def policy_improvement(self, show=False):
        state_nums = self.env.state_nums
        if show:
            print("策略提升前")
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    # 为了输出美观,保持输出6个字符
                    print(f"{ self.pi[i * self.env.ncol + j] }", end=" ")
                print()
        for s in range(state_nums):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state])
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
            # print(f"state: {s}  pi: {self.pi[s]}")
        if show:
            print("策略提升后")
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    # 为了输出美观,保持输出6个字符
                    print(f"{ self.pi[i * self.env.ncol + j] }", end=" ")
                print()
        # print("策略提升完成\n")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        # start_time = time.perf_counter()   # 计算收敛时间

        while 1:
            # print("===================策略迭代===================")
            # print(f"价值函数：{self.v}")
            # print(f"pi：{self.pi}")

            self.policy_evaluation(show=False)
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement(show=False)
            flag = True
            # 判断决策的行动是否相同。
            for i in range(self.env.state_nums):
                action_same_flag = True
                for j in range(4):
                    if (old_pi[i][j] > 0 and new_pi[i][j] > 0) or (old_pi[i][j] == 0 and new_pi[i][j] == 0):
                        pass
                    else:
                        action_same_flag = False
                        break
                if action_same_flag is False:
                    flag = False
                    break
            if flag:
                break

                # if old_pi == new_pi:
                # break
        # 计算收敛时间
        # end_time = time.perf_counter()
        # convergence_time = (end_time - start_time) * 1000
        # print(f"策略迭代收敛时间：{convergence_time} 毫秒")
        # return convergence_time

    # def policy_evaluation(self):  # 策略评估
    #     cnt = 1  # 计数器
    #     while 1:
    #         max_diff = 0
    #         # 初始化值函数
    #         new_v = [0] * self.env.ncol * self.env.nrow
    #         for s in range(self.env.ncol * self.env.nrow):
    #             qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
    #             for a in range(4):
    #                 qsa = 0
    #                 for res in self.env.P[s][a]:
    #                     p, next_state, r, done = res
    #                     qsa += p * (r + self.gamma *
    #                                 self.v[next_state] * (1 - done))
    #                     # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
    #                 qsa_list.append(self.pi[s][a] * qsa)
    #             new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
    #             max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
    #         self.v = new_v
    #         if max_diff < self.theta:
    #             break  # 满足收敛条件,退出评估迭代
    #         cnt += 1
    #     print("策略评估进行%d轮后完成" % cnt)

    # def policy_improvement(self):  # 策略提升
    #     for s in range(self.env.nrow * self.env.ncol):
    #         qsa_list = []
    #         for a in range(4):
    #             qsa = 0
    #             for res in self.env.P[s][a]:
    #                 p, next_state, r, done = res
    #                 qsa += p * (r + self.gamma * self.v[next_state] *
    #                             (1 - done))
    #             qsa_list.append(qsa)
    #         maxq = max(qsa_list)
    #         cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
    #         # 让这些动作均分概率
    #         self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
    #     print("策略提升完成")
    #     return self.pi


class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子
        self.init_values()

    # 初始化
    def init_values(self):
        state_nums = self.env.state_nums
        self.v = [0] * state_nums
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(state_nums)]

    # 价值迭代
    def value_iteration(self):
        state_nums = self.env.state_nums
        cnt = 0
        while True:
            # print("===================价值迭代===================")
            delta = 0
            for s in range(state_nums):
                v = self.v[s]
                max_qsa = float('-inf')
                for a in range(4):  # 假设有4个动作
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state])
                    max_qsa = max(max_qsa, qsa)
                self.v[s] = max_qsa
                delta = max(delta, abs(v - self.v[s]))
            # print(f"迭代误差： {delta}")
            cnt += 1
            if delta < self.theta:
                break
        self.compute_optimal_policy()

        return self.v

    # 计算最优策略
    def compute_optimal_policy(self):
        state_nums = self.env.state_nums
        for s in range(state_nums):
            qsa_list = []
            for a in range(4):  # 假设有4个动作
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state])
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


# 使用示例
# env = YourEnvironment()  # 你需要替换成你的环境类
# vi = ValueIteration(env, theta=0.0001, gamma=0.9)
# optimal_values = vi.value_iteration()
# print(optimal_values)
