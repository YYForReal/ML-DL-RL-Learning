from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym

experiment = Experiment(
  api_key="jxCKAgc1LK4bLO9pQIuerERSJ",
  project_name="rl-gym",
  workspace="yyforreal"
)

env = gym.make('Acrobot-v1', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'test')
env = CometLogger(env, experiment)

for x in range(20):
    observation, info = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
            

env.close() #Uploads video folder 'test' to Comet