BATCH_SIZE = 32
MAX_EPISODES = 50000
MAX_STEPS = 1000000
seed = 12345
# state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma
lr_actor = 0.001
lr_critic = 0.005
gamma = 0.99
hidden_dim = 256
# 设置加载点，加载模型
load_points = 0
env_name = "ALE/DemonAttack-ram-v5"
# env_name = "ALE/Pong-ram-v5"

# env_name = "ALE/DemonAttack-v5"
# env_name = "ALE/DonkeyKong-v5"
# env = gym.make("ALE/DonkeyKong-v5",render_mode="rgb_array",obs_type="ram")

target_update = 10

target_update = 5 # 每隔target步更新一次DQN的目标网络
buffer_size = 100000


n_updates_critic = 1 # 每n步更新一次critic
m_updates_actor = 4 # 每m步更新一次actor

mode = "AC"



hyperparameters = {
    "MAX_EPISODES": MAX_EPISODES,
    "seed":seed,
    "lr_actor": lr_actor,
    "lr_critic": lr_critic,
    "gamma": gamma,
    "hidden_dim": hidden_dim,
    "BATCH_SIZE":BATCH_SIZE,
    "load_points":load_points,
    "env_name":env_name,
    "target_update":target_update,
    "m_updates_actor":m_updates_actor,
    "n_updates_critic":n_updates_critic,
    "buffer_size":buffer_size,
    "mode":mode
}
