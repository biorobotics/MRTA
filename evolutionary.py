from env import MultiAgentEnv
from params import get_params
from geneticalgorithm import geneticalgorithm as ga
import numpy as np

params = get_params()
env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])

env_type = [0, 1, 2, 3]

# the function that we want to maximize
def f(x):
    x = x.reshape([4,3])
    x /= x.sum(axis=0)
    return - env.getReward(x, env_type)


varbound=np.array([[0,1]]*12)
model=ga(function=f,dimension=12,variable_type='real',variable_boundaries=varbound)

model.run()