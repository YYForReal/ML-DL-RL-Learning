from agent import MonteCarloAgent, SARSAgent, QLearningAgent
from env import Envi

from util import plot_value_function, plot_q_values, draw_animation
import wandb
import numpy as np


def main():
    env = Envi()
    wandb.init(project="MC-TD-S21", name="agent_experiment3")

    # 1. 使用蒙特卡洛策略的Agent
    # mc_agent = MonteCarloAgent(env, epsilon=0.1)
    # # 训练与测试
    # mc_agent.train(num_episodes=10000)
    # test_agent(mc_agent, env, "Monte Carlo")
    # plot_q_values(mc_agent.q_values)
    # 可视化
    # draw_animation(mc_agent, "Monte Carlo.gif")

    # 2. 使用SARSA策略的Agent
    # sarsa_agent = SARSAgent(env)
    # sarsa_agent.train(num_episodes=100000)
    # # test_agent(sarsa_agent, env, "SARSA")
    # plot_q_values(sarsa_agent.q_values, canshow=True, agent_name="SARSA")

    # 3. 使用Q-learning策略的Agent
    # qlearning_agent = QLearningAgent(env)
    # qlearning_agent.train(num_episodes=100000)
    # test_agent(qlearning_agent, env, "QLearning")
    # plot_q_values(qlearning_agent.q_values,
    #               canshow=True, agent_name="QLearning")

    # 4. 记录训练过程中的win_rate
    env1 = Envi()
    env2 = Envi()
    env3 = Envi()

    mc_agent = MonteCarloAgent(env1)
    sarsa_agent = QLearningAgent(env2)
    qlearning_agent = QLearningAgent(env3)

    # 5. 计算Sarsa与MC的均方差
    def mean_sqr(q1, q2):
        return np.sum(np.square(q1 - q2))

    for i in range(100):
        mc_agent.train(num_episodes=1000, canshow=False)
        sarsa_agent.train(num_episodes=1000, canshow=False)
        qlearning_agent.train(num_episodes=1000, canshow=False)

        win_rate1 = test_agent(mc_agent, env1, "MonteCarlo")
        win_rate2 = test_agent(sarsa_agent, env2, "SARSA")
        win_rate3 = test_agent(qlearning_agent, env3, "QLearning")

        # 使用wandb记录训练过程中的win_rate
        # 计算均方差
        q_values_sarsa = sarsa_agent.q_values
        q_values_mc = mc_agent.q_values
        q_values_qlearning = qlearning_agent.q_values
        # Calculate the mean squared difference
        mse_sarsa_vs_mc = mean_sqr(q_values_sarsa, q_values_mc)
        mse_sarsa_vs_qlearning = mean_sqr(q_values_sarsa, q_values_qlearning)
        mse_qlearning_vs_mc = mean_sqr(q_values_qlearning, q_values_mc)
        wandb.log({
            "mc_agent win_rate": win_rate1,
            "sarsa_agent win_rate": win_rate2,
            "qlearning_agent win_rate": win_rate3,
            "mse_sarsa_vs_mc": mse_sarsa_vs_mc,
            "mse_sarsa_vs_qlearning": mse_sarsa_vs_qlearning,
            "mse_qlearning_vs_mc": mse_qlearning_vs_mc,
        })

    # Assuming q_values_sarsa and q_values_mc are the Q-values matrices for Sarsa and MC agents
    # Train the agents first (use your actual training code)
    # sarsa_agent.train(num_episodes=1000, canshow=False)
    # mc_agent.train(num_episodes=1000, canshow=False)

    # Print or log the result
    print(
        f"Mean Squared Difference between Sarsa and MC Q-values: {mse_sarsa_vs_mc}")


def test_agent(agent, env, agent_name):
    total_rewards = 0
    num_episodes = 1000
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
    # print(f"{agent_name} Agent - Win Rate over {num_episodes} episodes: {win_rate}")
    # print("env.result_counts_description:", env.result_counts_description)
    # print("env.result_counts:", env.result_counts)
    return win_rate


if __name__ == "__main__":
    main()
