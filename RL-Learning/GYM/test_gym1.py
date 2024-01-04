import gym

env = gym.make('CartPole-v1')
observation = env.reset()
env.render_mode = "human"
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(f"{observation}  info {info} ")
    if truncated:
        print("truncated")
        break
    if done:
        print("done")
        break

env.close()
