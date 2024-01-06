from env import env ,experiment
from config import MAX_EPISODES,lr_actor,lr_critic,gamma,MAX_STEPS,BATCH_SIZE,env_name,target_update,seed,buffer_size,load_points

from models import ActorCriticAgent,DQNAgent
import os
import csv

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

csv_file_path = os.path.join(os.path.dirname(__file__), 'results', f'{env_name.split("/")[-1]}-{BATCH_SIZE}-{target_update}-results.csv')
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Episode', 'Reward', 'Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


print(STATE_DIM,ACTION_DIM)
# agent = ActorCriticAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim = STATE_DIM * 2,
#                          lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma)
agent = DQNAgent(env=env,buffer_size=buffer_size,target_update=target_update)
# 设置加载点，加载模型
load_path = os.path.join(os.path.dirname(__file__), "checkpoints",f"{env_name}-DQNAgent-{BATCH_SIZE}-{target_update}-{load_points}.pt")
if os.path.exists(load_path):
    agent.load_model(load_path)
    print("load model successful")


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    greedy_eps = 0.5 # 初始的随机概率
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Episode', 'Reward', 'Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for episode in range(max_episodes):
            observation,_ = env.reset(seed=seed+episode) # 重置游戏环境
            episode_reward = 0
            greedy_eps = max(0.05,greedy_eps*0.999)  # 随着游戏局数递减
            agent_loss = None
            for step in range(max_steps):
                # 动作选择
                action,probs = agent.select_action(observation,eps=greedy_eps)
                # 新版gym返回5个状态值
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # 存储到缓存区
                agent.replay_buffer.push(observation, action, reward, next_observation, done)
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    agent_loss = agent.update(batch_size) 

                if done or step == max_steps-1:
                    episode_rewards.append(episode_reward)
                    break
                
                observation = next_observation
            print(f"Episode {episode}: {episode_rewards[-1]} loss: {agent_loss} greedy_eps {greedy_eps}")

            # 使用comet_ml记录
            experiment.log_metric("loss", agent_loss,epoch=episode + load_points)
            experiment.log_metric("reward", episode_reward,epoch=episode + load_points)

            writer.writerow({'Episode': episode + load_points, 'Reward': episode_reward, 'Loss': agent_loss})
            if (episode + 1) % 10 == 0:
                csvfile.flush()  # Flush the buffer to ensure data is written to disk
            # 记录过往10的平均分数
            if len(episode_rewards) > 10:
                mean_reward = sum(episode_rewards[-10:]) / 10
                print(f"Mean reward of last 10 episodes: {mean_reward}")
                experiment.log_metric("mean_reward", mean_reward,epoch=episode + load_points)


            # 每隔1000次迭代保存模型
            if (episode + load_points) % 1000 == 0:
                # os拼接路径
                path = os.path.join(os.path.dirname(__file__), 'checkpoints', f'{env_name}-DQNAgent-{BATCH_SIZE}-{target_update}-{episode + load_points}.pt')
                agent.save_model(path)

    return episode_rewards



def train(env):
    print("action space : ", env.action_space)
    for episode in range(MAX_EPISODES):
        observation, info = env.reset()
        truncated = False
        terminated = False
        print(observation)
        print(type(observation))
        while not (truncated or terminated):
            # action = env.action_space.sample()
            print("observation",observation)
            action,_ = agent.select_action(observation)
            print("choose action",action)
            new_observation, reward, terminated, truncated, info = env.step(action)
            print("observation",observation)
            agent.update([observation], [action], [reward], [new_observation], [terminated])
            env.render()
            observation = new_observation

    env.close() #Uploads video folder 'test' to Comet


# train(env)

mini_batch_train(env,agent,MAX_EPISODES,MAX_STEPS,BATCH_SIZE)