from agent import MonteCarloAgent, SARSAgent, QLearningAgent
from env import Envi

from util import plot_value_function, plot_q_values, draw_animation


def main():
    env = Envi()

    # 1. 使用蒙特卡洛策略的Agent
    mc_agent = MonteCarloAgent(env, epsilon=0.1)
    # 训练与测试
    mc_agent.train(num_episodes=10000)
    # test_agent(mc_agent, env, "Monte Carlo")
    # plot_q_values(mc_agent.q_values)
    # 可视化:启动绘制时不需要再额外训练了.
    draw_animation(mc_agent, "Monte Carlo.gif", "Monte Carlo")

    # 2. 使用SARSA策略的Agent
    sarsa_agent = SARSAgent(env)
    # sarsa_agent.train(num_episodes=100000)
    # test_agent(sarsa_agent, env, "SARSA")
    # plot_q_values(sarsa_agent.q_values, canshow=True, agent_name="SARSA")
    draw_animation(sarsa_agent, "SARSA.gif", "SARSA")

    # 3. 使用Q-learning策略的Agent
    qlearning_agent = QLearningAgent(env)
    # qlearning_agent.train(num_episodes=100000)
    # test_agent(qlearning_agent, env, "QLearning")
    # plot_q_values(qlearning_agent.q_values,
    #               canshow=True, agent_name="QLearning")
    draw_animation(qlearning_agent, "QLearning.gif", "QLearning")

    # 4. 记录训练过程中的win_rate
    # qlearning_agent = QLearningAgent(env)
    # for i in range(100):
    #     qlearning_agent.train(num_episodes=1000)
    #     win_rate = test_agent(qlearning_agent, env, "QLearning")
    #     # 使用wandb记录训练过程中的win_rate
    #     # wandb.log({"win_rate": win_rate})


def test_agent(agent, env, agent_name):
    total_rewards = 0
    num_episodes = 10000
    win_counts = 0
    env.result_counts = [0]*5  # 重置游戏结果记录
    for _ in range(num_episodes):
        state, reward, done = env.reset()
        # print("init state:", state)

        while not done:
            action = agent.choose_best_action(state)  # 使用贪婪策略测试
            next_state, reward, done = env.step(action)
            state = next_state
            total_rewards += reward
            if done:
                if reward == 1:
                    win_counts += 1
                break
            # print("total_rewards:", total_rewards)

    win_rate = win_counts / num_episodes
    print(f"{agent_name} Agent - Win Rate over {num_episodes} episodes: {win_rate}")
    print("env.result_counts_description:", env.result_counts_description)
    print("env.result_counts:", env.result_counts)
    return win_rate


if __name__ == "__main__":
    main()
