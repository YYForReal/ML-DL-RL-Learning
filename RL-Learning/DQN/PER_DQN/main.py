# import gym
import gymnasium as gym

import os
import numpy as np
import random
import torch
from network import MLP
from agent import PER_DQN
from buffer import ReplayTree
from utils import Config, plot_rewards
from gymnasium.wrappers import RecordVideo

def episode_trigger(episode):
    if (episode) % 10 == 0:
        return True
    return False

def all_seed(env,seed = 1):
    ''' 万能的seed函数
    '''
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):

    # before_training = "before_training.mp4"
    env = gym.make(cfg.env_name,render_mode="rgb_array") # 创建环境
    # video = VideoRecorder(env, before_training)
    env = RecordVideo(
        env, f"videos", episode_trigger=episode_trigger, name_prefix=cfg.env_name)



    if cfg.seed !=0:
        all_seed(env,seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")

    cfg.n_actions = env.action_space.n  ## set the env action space
    model = MLP(n_states, n_actions, hidden_dim = cfg.hidden_dim) # 创建模型
    memory = ReplayTree(cfg.buffer_size)
    agent = PER_DQN(model,memory,cfg)
    return env,agent

def train(cfg, env, agent):
    ''' 训练
    '''
    print("开始训练！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        state = state[0]   # state是一个tuple、第一个是state，第二个是dtype
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 选择动作
            # next_state, reward, done, _= env.step(action)  # 更新环境，返回transition
            next_state, reward, done,terminated, _ = env.step(action)  # 更新环境，返回transition
            done = done | terminated

            ## PER DQN 特有的内容
            policy_val = agent.policy_net(torch.tensor(state, device = cfg.device))[action]
            target_val = agent.target_net(torch.tensor(next_state, device = cfg.device))

            if done:
                error = abs(policy_val - reward)
            else:
                error = abs(policy_val - reward - cfg.gamma * torch.max(target_val))

            agent.memory.push(error.cpu().detach().numpy(), (state, action, reward,
                            next_state, done))   # 保存transition
            
            agent.update()  # 更新智能体
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}")
    print("完成训练！")
    env.close()
    return {'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_steps):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done,terminated, _ = env.step(action)  # 更新环境，返回transition
            done = done | terminated
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}



# 获取参数
cfg = Config() 
# 训练
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent)
 
plot_rewards(res_dic['rewards'], cfg, tag="train")  
# 测试
res_dic = test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果