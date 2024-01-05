BATCH_SIZE = 64
MAX_EPISODES = 10000
MAX_STEPS = 5000
seed = 12345
# state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma
lr_actor = 0.001
lr_critic = 0.005
gamma = 0.99
hidden_dim = 128
# 设置加载点，加载模型
load_points = 0
env_name = "ALE/DemonAttack-v5"

hyperparameters = {
    "MAX_EPISODES": MAX_EPISODES,
    "seed":seed,
    "lr_actor": lr_actor,
    "lr_critic": lr_critic,
    "gamma": gamma,
    "hidden_dim": hidden_dim,
    "BATCH_SIZE":BATCH_SIZE,
    "load_points":load_points,
    "env_name":env_name
}
