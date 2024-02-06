import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import numpy as np
import random

class PolicyNet(paddle.nn.Layer):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = paddle.nn.Linear(state_dim, hidden_dim)
        self.fc2 = paddle.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))


class ValueNet(paddle.nn.Layer):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = paddle.nn.Linear(state_dim, hidden_dim)
        self.fc2 = paddle.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# 计算优势advantage
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return paddle.to_tensor(advantage_list, dtype='float32')



class PPO:
    ''' PPO-clip,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,lmbda,
                 epochs, eps, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)
        self.actor_optimizer = paddle.optimizer.Adam(parameters=self.actor.parameters(),
                                                learning_rate=actor_lr)
        self.critic_optimizer = paddle.optimizer.Adam(parameters=self.critic.parameters(),
                                                 learning_rate=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用于训练轮数
        self.eps = eps  # PPO中截断范围的参数
        

    def take_action(self, state):
        state = paddle.to_tensor(state, dtype='float32')
        probs = self.actor(state)
        action_dist = paddle.distribution.Categorical(probs)
        action = action_dist.sample([1])
        return action.numpy()[0]


    def update(self, transition_dict):
        states = paddle.to_tensor(transition_dict['states'],dtype='float32')
        actions = paddle.to_tensor(transition_dict['actions']).reshape([-1, 1])
        rewards = paddle.to_tensor(transition_dict['rewards'],dtype='float32').reshape([-1, 1])
        next_states = paddle.to_tensor(transition_dict['next_states'],dtype='float32')
        dones = paddle.to_tensor(transition_dict['dones'],dtype='float32').reshape([-1, 1])
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)

        advantage = compute_advantage(self.gamma, self.lmbda,td_delta)
        old_log_probs = paddle.log(self.actor(states).gather(axis=1,index=actions)).detach()

        for _ in range(self.epochs):
            log_probs = paddle.log(self.actor(states).gather(axis=1, index=actions))
            ratio = paddle.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = paddle.clip(ratio, 1 - self.eps,1 + self.eps) * advantage  # 截断
            actor_loss = paddle.mean(-paddle.minimum(surr1, surr2))  # PPO损失函数
            critic_loss = paddle.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.clear_grad()
            self.critic_optimizer.clear_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self):
        paddle.save(self.actor.state_dict(),'net.pdparams')

    def load(self):
        layer_state_dict = paddle.load("net.pdparams")
        self.actor.set_state_dict(layer_state_dict)  


actor_lr = 1e-3 #策略网络的学习率
critic_lr = 1e-2 #价值网络的学习率
num_episodes = 100 # 训练的episode,不宜训练太长，否则性能下降
hidden_dim = 128 #网络隐藏层
gamma = 0.98 # 折扣因子
lmbda = 0.95 # 优势计算中的参数
epochs = 10  #每次更新时ppo的更新次数
eps = 0.2 # PPO中截断范围的参数


env_name = 'CartPole-v0'
env = gym.make(env_name)
# env.seed(100)
# paddle.seed(100)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    maxre=0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                
                # 保存最大epoisde奖励的参数
                if maxre<episode_return:
                    maxre=episode_return
                    agent.save()

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma)
return_list = train_on_policy_agent(env, ppo_agent, num_episodes)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

episodes_list = list(range(len(return_list)))
mv_return = moving_average(return_list, 19)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()



# actor=PolicyNet(4,128,2)
# layer_state_dict = paddle.load("net.pdparams")
# actor.set_state_dict(layer_state_dict)


# env=gym.make('CartPole-v0')

# state=env.reset()
# frames = []
# for i in range(200):
    
    
#     state=paddle.to_tensor(state,dtype='float32')
#     action =actor(state).numpy()
#     #action=action.numpy()[0]
#     #print(action)
#     next_state,reward,done,_=env.step(np.argmax(action))
#     if i%10==0:
#         print(i,"   ",reward,done)
#     state=next_state

# env.close()


# 可视化
# actor=PolicyNet(4,128,2)
# layer_state_dict = paddle.load("net.pdparams")
# actor.set_state_dict(layer_state_dict)

# def save_frames_as_gif(frames, filename):

#     #Mess with this to change frame size
#     plt.figure(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100), dpi=300)

#     patch = plt.imshow(frames[0])
#     plt.axis('off')

#     def animate(i):
#         patch.set_data(frames[i])

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#     anim.save(filename, writer='pillow', fps=60)

# env=gym.make('CartPole-v0')

# state=env.reset()
# frames = []
# for i in range(200):
#     #print(env.render(mode="rgb_array"))
#     frames.append(env.render(mode="rgb_array"))
#     state=paddle.to_tensor(state,dtype='float32')
#     action =actor(state).numpy()
#     #action=action.numpy()[0]
#     #print(action)
#     next_state,reward,done,_=env.step(np.argmax(action))
#     if i%50==0:
#         print(i,"   ",reward,done)
#     state=next_state

# save_frames_as_gif(frames, filename="CartPole.gif")
    
# env.close()