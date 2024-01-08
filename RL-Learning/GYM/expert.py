import gymnasium as gym
import os
from networks import ReplayBuffer
from pynput import keyboard

env_name = "ALE/DemonAttack-ram-v5"
filter_name = env_name.split('/')[-1]
replay_buffer = ReplayBuffer(1000000)
load_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"{filter_name}.npy")


if os.path.exists(load_path):
    print("load replay buffer", load_path)

    replay_buffer.load(load_path)
    print("size: ", len(replay_buffer))


observation = None
terminated = False
truncated = False
last_key = 'w'
action = 1
is_press_w = False
is_press_d = False
is_press_a = False

def on_key_press(key):
    global action, is_press_w, is_press_d, is_press_a
    try:
        if key.char == 'w':
            is_press_w = True  # Simulate continuous fire while 'w' is held
            action = 1
        elif key.char == 'd':
            is_press_d = True  # Simulate pressing 'd'
            action = 4
        elif key.char == 'a':
            is_press_a = True  # Simulate pressing 'a'
            action = 5
    except AttributeError:
        pass

def on_key_release(key):
    global observation, terminated, truncated,last_key,action
    try:
        action = 1
        print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        pass


def human_train(env):
    global observation, terminated, truncated,action
    observation, info = env.reset()
    print("action space : ", env.action_space)
    try:
        with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
            for episode in range(1):
                observation, info = env.reset()
                truncated = False
                terminated = False
                while not (truncated or terminated):
                    old_action = action
                    print("action", old_action)
                    new_observation, reward, terminated, truncated, info = env.step(old_action)
                    replay_buffer.push(observation, old_action, reward, new_observation, terminated)
                    env.render()
                    observation = new_observation
    except Exception as e:
        print("error", e)
    ready = int(input("ready to save (0,1) ? "))
    if ready == 1:
        replay_buffer.save(load_path)
        print("save replay buffer")
        print("size: ", len(replay_buffer))
    env.close()  # Uploads video folder 'test' to Comet


mode = "human"
render_mode = "human"
obs_type = "ram"
env = gym.make(env_name, render_mode=render_mode, obs_type=obs_type)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
print(STATE_DIM, ACTION_DIM)
human_train(env)
