from env import make_env
from config import MAX_EPISODES,lr_actor,lr_critic,gamma,MAX_STEPS,BATCH_SIZE,env_name,target_update,m_updates_actor,n_updates_critic,load_points,hidden_dim,hyperparameters

from models import ActorCriticAgent,DQNAgent
import os
import wandb

os.environ["WANDB_API_KEY"] = "44d22c3006abd6799112a392f847ac52b285ad31"


# 设置加载点，加载模型
base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    greedy_eps = 0.5
    max_reward = -22
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
        max_reward = max(episode_reward,max_reward)


        wandb.log({"loss":agent_loss,"reward":episode_reward,"max_reward":max_reward},step=episode + load_points)

        # 使用comet_ml记录
        # experiment.log_metric("loss", agent_loss,epoch=episode + load_points)
        # experiment.log_metric("reward", episode_reward,epoch=episode + load_points)

        # 每隔200次迭代保存模型
        if episode % 1000 == 0:
            # os拼接路径
            agent.save_model(base_path,episode + load_points)

    return episode_rewards


def train_single(env,agent,mode,extra_title=""):
    wandb.init(project="gym",name=f"{env_name}-{mode}-{extra_title}",config=hyperparameters)
    mini_batch_train(env,agent,MAX_EPISODES,MAX_STEPS,BATCH_SIZE)
    wandb.finish()
    
    
mode = "ActorCritic"
env = make_env(env_name)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
print(STATE_DIM,ACTION_DIM)
# # 设置加载点，加载模型
# base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')

# if os.path.exists(base_path):
#     agent.load_model(base_path)


m_updates_actors = [1]
n_updates_critics = [1]

for i in range(len(m_updates_actors)):
    hyperparameters["m_updates_actor"] = m_updates_actors[i]
    hyperparameters["n_updates_critic"] = n_updates_critics[i]
    agent = ActorCriticAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim = hidden_dim,
                            lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma,
                            m_updates_actor=m_updates_actors[i],n_updates_critic=n_updates_critics[i])

    train_single(env=env,agent=agent,mode=mode,extra_title=f"m-{m_updates_actors[i]}-n-{n_updates_critics[i]}")
    
    print("over",i)