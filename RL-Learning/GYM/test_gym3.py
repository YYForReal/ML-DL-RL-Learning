from env import env ,experiment
from config import MAX_EPISODES,lr_actor,lr_critic,gamma,MAX_STEPS,BATCH_SIZE,load_points
import numpy as np
from models import ActorCriticAgent,DQNAgent
import os
# 使用cemet_ml



STATE_DIM = env.observation_space.shape
ACTION_DIM = env.action_space.n
print(STATE_DIM,ACTION_DIM)
# input("----")
# agent = ActorCriticAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim = STATE_DIM * 2,
#                          lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma)
agent = DQNAgent(env=env)
load_path = os.path.join(os.path.dirname(__file__), "checkpoints",f"DQNAgent-mask-{load_points}.pt")
if os.path.exists(load_path):
    agent.load_model(load_path)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    greedy_eps = 0.5

    for episode in range(max_episodes):
        observation,_ = env.reset()
        episode_reward = 0
        greedy_eps = max(0.01,greedy_eps*0.999) 
        agent_loss = None
        for step in range(max_steps):
            # print("observation",observation)
            # print('type obs',type(observation))
            mask_action_space = np.array( [2,3,4,6,7])
            action,probs = agent.select_action(observation,eps=greedy_eps,mask_action_space=mask_action_space)

            # next_observation, reward, done, _ = env.step(action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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
        experiment.log_metric("loss", agent_loss,epoch=episode+load_points)
        experiment.log_metric("reward", episode_reward,epoch=episode+load_points)

        # 每隔100次迭代保存模型
        if episode % 100 == 0:
            # os拼接路径
            path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'DQNAgent-' + str(episode + load_points) + '.pt')
            agent.save_model(path)
            print("save model to ",path)

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