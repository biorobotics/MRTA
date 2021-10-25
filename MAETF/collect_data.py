import numpy as np
import matplotlib.pyplot as plt

from .ergodicity import ErgCalculator
from .agent import xyAgent
from .team import optimize_team_erg
from scipy.special import softmax
from .simulator import MultiAgentEnv
import multiprocessing
import time

n_data = 5000
# erg_list = []
# assignment_list = []
# env_list = []

manager = multiprocessing.Manager()
assignment_list = manager.list()
erg_list = manager.list()
env_list = manager.list()
l = multiprocessing.Lock()

def get_random_alloc(env):
    cont_alloc = False
    if cont_alloc:
        assignment = np.random.uniform(-1, 1, [env.n_types_agents, env.n_num_grids])
        assignment = softmax(assignment, axis=1)
    else:
        assignment = np.random.randint(21, size=[env.n_types_agents, env.n_num_grids])
        # print(assignment)
    return assignment

def get_reward(env, terrain):
    assignment = get_random_alloc(env)
    reward, reward_list = env.get_integer_reward(assignment, terrain)
    return assignment, reward_list

def data_collection_worker():
    env = MultiAgentEnv()
    terrain = np.zeros((50, 50), dtype=np.int32)
    terrain[0:25, 0:25] = 1
    terrain[0:25, 25:] = 2
    terrain[25:, 25:] = 3
    np.random.seed()

    while len(env_list) < n_data:
        assignment, reward_list = get_reward(env, terrain)
        l.acquire()
        assignment_list.append(assignment)
        erg_list.append(reward_list)
        env_list.append([0, 1, 2, 3])
        # print(assignment)
        # print(reward_list)
        l.release()

        if len(env_list) % 100 == 0:

            print(f"collected {len(env_list)} data")
            np.save("MAETF/data/assignments_large", assignment_list)
            np.save("MAETF/data/ergs_large", erg_list)
            np.save("MAETF/data/env_labels_large", env_list)


process_list = []
for i in range(3):
    p = multiprocessing.Process(target=data_collection_worker, args=[])
    p.start()
    process_list.append(p)
    time.sleep(0.01)

for process in process_list:
    process.join()



