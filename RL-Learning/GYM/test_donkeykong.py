# import gymnasium as gym
# env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()

import gymnasium as gym
env = gym.make("ALE/DonkeyKong-v5", render_mode="human",obs_type="ram")
observation, info = env.reset(seed=42)
for _ in range(100):
   env.render()
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   print(observation)
   if terminated or truncated:
      observation, info = env.reset()

env.close()