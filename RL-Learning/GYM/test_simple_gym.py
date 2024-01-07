from env import make_env
from config import MAX_EPISODES,lr_actor,lr_critic,gamma,MAX_STEPS,BATCH_SIZE,env_name,target_update,hyperparameters,env_name,load_points

from models import ActorCriticAgent,DQNAgent
import os
import wandb


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    greedy_eps = 0.5
    max_reward = 0
    for episode in range(max_episodes):
        observation,_ = env.reset()
        episode_reward = 0
        greedy_eps = max(0.05,greedy_eps*0.999) 
        agent_loss = None
        for step in range(max_steps):
            mask_action_space = None
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
        max_reward = max(episode_reward,max_reward)
        # 使用comet_ml记录
        # experiment.log_metric("loss", agent_loss,epoch=episode + load_points)
        # experiment.log_metric("reward", episode_reward,epoch=episode + load_points)
        # experiment.log_metric("max_reward", max_reward,epoch=episode + load_points)

        wandb.log({"loss":agent_loss,"reward":episode_reward,"max_reward":max_reward},step=episode + load_points)

        # 每隔200次迭代保存模型
        if episode % 1000 == 0:
            # os拼接路径
            path = os.path.join(os.path.dirname(__file__), 'checkpoints', f'{env_name}-DQNAgent-{BATCH_SIZE}-{target_update}-{episode + load_points}.pt')
            agent.save_model(path)

        # os拼接路径
        path = os.path.join(os.path.dirname(__file__), 'checkpoints', f'{env_name}-DQNAgent-{BATCH_SIZE}-{target_update}-final.pt')
        agent.save_model(path)


    return episode_rewards



def train_single(env_name,agent,mode,title):
    wandb.init(project="gym",name=f"{env_name}-{mode}",config=hyperparameters)
    mini_batch_train(env,agent,MAX_EPISODES,MAX_STEPS,BATCH_SIZE)
    wandb.finish()

mode = "DQNAgent"
env = make_env(env_name)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
print(STATE_DIM,ACTION_DIM)
agent = DQNAgent(env=env,target_update=target_update)






