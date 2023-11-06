from env import CliffWalkingEnv, GridWorld
from policy import PolicyIteration, ValueIteration
from utils import print_agent, print_v
import time

ncol = 4
nrow = 4
env = GridWorld(ncol, nrow)
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 1
"""
正常测试：
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, [0, ncol*nrow-1], action_meaning)


agent2 = ValueIteration(env, theta, gamma)
agent2.value_iteration()
# print_agent(agent2, [0, ncol*nrow-1], action_meaning)
# print_v(agent2.v, env.nrow, env.ncol)

"""


aver_policy = 0.0
aver_value = 0.0
test_num = 100
# 计算收敛时间
for i in range(test_num):

    agent = PolicyIteration(env, theta, gamma)
    agent2 = ValueIteration(env, theta, gamma)

    # 计时
    start_time = time.perf_counter()
    agent.policy_iteration()
    end_time = time.perf_counter()
    convergence_time = (end_time - start_time) * 1000
    print(f"策略迭代收敛时间：{convergence_time} 毫秒")
    aver_policy += convergence_time
    # 计时
    start_time = time.perf_counter()
    agent2.value_iteration()
    end_time = time.perf_counter()
    convergence_time = (end_time - start_time) * 1000
    print(f"价值迭代收敛时间：{convergence_time} 毫秒")
    aver_value += convergence_time

print(f"策略迭代平均收敛时间：{aver_policy/test_num} 毫秒")
print(f"价值迭代平均收敛时间：{aver_value/test_num} 毫秒")
