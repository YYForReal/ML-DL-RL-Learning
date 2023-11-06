from env import CliffWalkingEnv, GridWorld
from policy import PolicyIteration, ValueIteration
from utils import print_agent, print_v


ncol = 4
nrow = 4
env = GridWorld(ncol, nrow)
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 1
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, [0, ncol*nrow-1], action_meaning)


agent2 = ValueIteration(env, theta, gamma)
agent2.value_iteration()
print_agent(agent2, [0, ncol*nrow-1], action_meaning)
# print_v(agent2.v, env.nrow, env.ncol)
