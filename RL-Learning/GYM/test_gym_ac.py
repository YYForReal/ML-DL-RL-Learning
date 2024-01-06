from env import env ,experiment
from config import MAX_EPISODES,lr_actor,lr_critic,gamma,MAX_STEPS,BATCH_SIZE,env_name,target_update,m_updates_actor,n_updates_critic,load_points,hidden_dim

from models import ActorCriticAgent,DQNAgent
import os

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
print(STATE_DIM,ACTION_DIM)
agent = ActorCriticAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim = hidden_dim,
                         lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma,
                         m_updates_actor=m_updates_actor,n_updates_critic=n_updates_critic)
# 设置加载点，加载模型
base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')

if os.path.exists(base_path):
    agent.load_model(base_path)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    greedy_eps = 0.5

    for episode in range(max_episodes):
        observation,_ = env.reset()
        episode_reward = 0
        greedy_eps = max(0.05,greedy_eps*0.999) 
        agent_loss = None
        for step in range(max_steps):
            # print("observation",observation)
            # print('type obs',type(observation))

            action,probs = agent.select_action(observation,eps=greedy_eps)
            # next_observation, reward, done, _ = env.step(action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            # states, actions, rewards, next_states, dones
            agent.update(observation, action, reward, next_observation, done)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                break
            
            observation = next_observation
        print(f"Episode {episode}: {episode_rewards[-1]}  greedy_eps {greedy_eps}")

        # 使用comet_ml记录
        # experiment.log_metric("loss", agent_loss,epoch=episode + load_points)
        experiment.log_metric("reward", episode_reward,epoch=episode + load_points)

        # 每隔200次迭代保存模型
        if episode % 1000 == 0:
            # os拼接路径
            agent.save_model(base_path,episode + load_points)

    return episode_rewards



def train(env):
    print("action space : ", env.action_space)
    for episode in range(MAX_EPISODES):
        observation, info = env.reset()
        truncated = False
        terminated = False
        while not (truncated or terminated):
            # action = env.action_space.sample()
            action,_ = agent.select_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            agent.update([observation], [action], [reward], [new_observation], [terminated])
            env.render()
            observation = new_observation

    env.close() #Uploads video folder 'test' to Comet


# train(env)

mini_batch_train(env,agent,MAX_EPISODES,MAX_STEPS,BATCH_SIZE)