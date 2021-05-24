# Author: Jiaheng Hu
# Sample a batch of env and alloc, test the reward net prediction
# For now, we are sampling data from the dataset

import numpy as np
from Networks.RewardNet import RewardNet
import torch
from params import get_params
from utils import int_to_onehot
from MAETF.simulator import MultiAgentEnv
import os
from train_simulation_reward import train_test_split, preprocess_data, load_data

params = get_params()

worker_device = torch.device("cuda:0")

if __name__ == '__main__':
    # 3*3
    net = RewardNet(params['n_agent_types'],
                    env_length=params['n_env_types'],
                    n_hidden_layers=5,
                    hidden_layer_size=256).to(worker_device)

    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])

    allocs, ergs, env_type = load_data(params)

    assert (allocs.shape[0] == ergs.shape[0] == env_type.shape[0])
    num_data = allocs.shape[0]

    ergs, allocs, env_type = preprocess_data(ergs, allocs, env_type)

    train_allocs, test_allocs = train_test_split(allocs, 0.8)
    train_erg, test_erg = train_test_split(ergs, 0.8)
    train_env, test_env = train_test_split(env_type, 0.8)

    loss_func = torch.nn.MSELoss()

    out_dir = params['reward_loc']
    # out_dir += '%s_nsamp:%d' % (params['data_method'], params['n_samples'])
    net.load_state_dict(torch.load(os.path.join(out_dir, "reward_weight")))
    net.eval()

    batch_size = 128
    for first in range(0, test_erg.shape[0], batch_size):
        alloc_batch = test_allocs[first:first + batch_size]
        targets = test_erg[first:first + batch_size]

        cur_batch_size = targets.shape[0]
        # env_vect = np.asarray([0, 1, 2, 3] * cur_batch_size)
        env_vect = test_env[first:first + cur_batch_size]
        # convert to onehot for further processing
        env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

        env_vect = torch.from_numpy(env_onehot).view(cur_batch_size, -1).float().to(worker_device)
        # convert numpy array to tensor in shape of input size
        alloc_batch = torch.from_numpy(alloc_batch).view(cur_batch_size, -1).float().to(worker_device)
        targets = torch.from_numpy(targets).float().reshape((-1, 1)).to(worker_device)

        prediction = net(alloc_batch, env_vect)
        loss = loss_func(prediction, targets)
        print(alloc_batch[:10])
        print(env_vect[:10])
        print(prediction[:10])
        print(targets[:10])
