import argparse
import matplotlib.pyplot as plt
import seaborn as sns

class Config():
    def __init__(self) -> None:
        # self.env_name = "CartPole-v1" # 环境名字
        # self.env_name = "CartPole-v1" # 环境名字
        self.env_name = 'Acrobot-v1'

        self.new_step_api = True # 是否用gym的新api
        self.wrapper = None 
        self.render = False 
        self.algo_name = "PER_DQN" # 算法名字
        self.mode = "train" # train or test
        self.seed = 0 # 随机种子
        self.device = "cpu" # device to use
        self.train_eps = 100 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        self.eval_eps = 10 # 评估的回合数
        self.eval_per_episode = 5 # 每个回合的评估次数
        self.max_steps = 2000 # 每个回合的最大步数
        self.load_checkpoint = False
        self.load_path = "tasks" # 加载模型的路径
        self.show_fig = False # 是否展示图片
        self.save_fig = True # 是否存储图片

        # 设置epsilon值
        self.epsilon_start = 0.95 # 起始的epsilon值
        self.epsilon_end = 0.01 # 终止的epsilon值
        self.epsilon_decay = 50 # 衰减率
        self.hidden_dim = 256 
        self.gamma = 0.95 
        self.lr = 0.0001 
        self.buffer_size = 100000 # 经验回放的buffer大小
        self.batch_size = 64 # batch size
        self.target_update = 4 # 目标网络更新频率
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()
