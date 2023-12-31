from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym
from config import hyperparameters,MAX_EPISODES,env_name

def episode_trigger_func(episode):
    print("choose episode ",episode)
    if episode  <= 1:
        return True
    if episode % (MAX_EPISODES // 10) == 0:
        return True
    return False

# 去除env_name的/,取右边内容
project_name = env_name.split('/')[-1]

experiment = Experiment(
  api_key="jxCKAgc1LK4bLO9pQIuerERSJ",
  project_name=project_name,
  workspace="gym"
)




# env = gym.make('Acrobot-v1', render_mode="rgb_array")
# 
# env = gym.make("ALE/DonkeyKong-v5",render_mode="rgb_array",obs_type="ram")
env = gym.make(env_name,render_mode="rgb_array",obs_type="ram")
env = gym.wrappers.RecordVideo(env, f'video-{env_name}')
env = CometLogger(env, experiment)


STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

hyperparameters['STATE_DIM'] = STATE_DIM
hyperparameters['ACTION_DIM'] = ACTION_DIM


print("env.observation_space",env.observation_space)
print("env.action_space",env.action_space)
experiment.log_parameters(hyperparameters)
