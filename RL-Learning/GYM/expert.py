import gymnasium as gym
import os
from networks import ReplayBuffer
from pynput import keyboard
from env import make_env

# env_name = "ALE/DemonAttack-v5"
# env_name = "ALE/DonkeyKong-v5"
env_name = "PongDeterministic-v4"

filter_name = env_name.split('/')[-1]
mode = "human"
render_mode = "human"
load_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"{filter_name}.npz")


def load_buffer(load_path):    
    replay_buffer = ReplayBuffer(1000000)
    print(load_path)
    if os.path.exists(load_path):
        print("load replay buffer", load_path)

        replay_buffer.load(load_path)
        print("size: ", len(replay_buffer))
    return replay_buffer

observation = None
terminated = False
truncated = False
last_key = 'w'
action = 1
is_press_w = False
is_press_d = False
is_press_a = False
is_press_s = False

# 射击  
def on_key_press1(key):
    global action, is_press_w, is_press_d, is_press_a
    try:
        if key.char == 'w':
            is_press_w = True  # w映射到开火动作
            action = 1
        elif key.char == 'd':
            is_press_d = True  # d右移动且开火
            action = 4
        elif key.char == 'a':
            is_press_a = True  # a左移动且开火
            action = 5
    except AttributeError:
        pass

def on_key_release1(key):
    global observation, terminated, truncated,last_key,action
    try:
        action = 1
        print(' key {0} pressed'.format(key.char))
    except AttributeError:
        pass


# 乒乓球
def on_key_press2(key):
    global action, is_press_w, is_press_d, is_press_a
    try:
        if key.char == 'w':
            is_press_w = True  # Simulate continuous fire while 'w' is held
            action = 2
        elif key.char == 's':
            is_press_s = True  # Simulate pressing 'a'
            action = 3
    except AttributeError:
        pass

def on_key_release2(key):
    global observation, terminated, truncated,last_key,action
    try:
        action = 0
        print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        pass

on_key_press = on_key_press2
on_key_release = on_key_release2


def human_train(env):
    # 设置全局的action变量，随着按键即时更新
    global observation, terminated, truncated,action
    replay_buffer = load_buffer(load_path)
    print("action space : ", env.action_space)
    try:
        # 持续监听键盘
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
    env.close()  





# obs_type = "ram"
# env = gym.make(env_name, render_mode=render_mode)
# env = gym.wrappers.AtariPreprocessing(env,  frame_skip=1,grayscale_obs=True,terminal_on_life_loss=True) 
env = make_env(env_name,mode,render_mode=render_mode,terminal_on_life_loss=True)


human_train(env)
