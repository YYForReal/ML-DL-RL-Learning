from comet_ml import Experiment, OfflineExperiment 
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym
from config import hyperparameters,MAX_EPISODES,env_name,BATCH_SIZE,target_update
from gymnasium.wrappers import FrameStack,FlattenObservation

def episode_trigger_func(episode):
    print("choose episode ",episode)
    if episode  <= 1:
        return True
    if episode % (MAX_EPISODES // 10) == 0:
        return True
    return False

# 去除env_name的/,取右边内容
project_name = env_name.split('/')[-1]


# offline_directory="/path/to/save/experiments"
experiment = OfflineExperiment(
  api_key="jxCKAgc1LK4bLO9pQIuerERSJ",
  project_name=project_name,
  workspace="gym",
  offline_directory="./experiments"
)




# env = gym.make('Acrobot-v1', render_mode="rgb_array")
# 
# env = gym.make("ALE/DonkeyKong-v5",render_mode="rgb_array",obs_type="ram")

env = gym.make(env_name,render_mode="rgb_array",obs_type="grayscale") # 转成单通道图像：维度128
print("STATE_DIM 1",env.observation_space.shape)

env = FrameStack(env, 1) # 堆叠4帧，可以学习到场景的方向,此时维度是(4,128)

print("STATE_DIM 2",env.observation_space.shape)
env = FlattenObservation(env) # 转为1维的shape 

STATE_DIM = env.observation_space.shape[0] # 此时state_dim = 512

print("STATE_DIM 3",env.observation_space.shape)

env = gym.wrappers.RecordVideo(env, f'video-{env_name}-{BATCH_SIZE}-{target_update}')
env = CometLogger(env, experiment)




ACTION_DIM = env.action_space.n

hyperparameters['STATE_DIM'] = STATE_DIM
hyperparameters['ACTION_DIM'] = ACTION_DIM


print("env.observation_space",env.observation_space)
print("env.action_space",env.action_space)
experiment.log_parameters(hyperparameters)
