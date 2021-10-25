'''
Author: Jiaheng Hu
Sample a batch of env and alloc, test the reward net prediction
For now, we are sampling data from the dataset
'''


import numpy as np
from Networks.RewardNet import RewardNet
import torch
from params import get_params
from utils import int_to_onehot, numpy_to_input_batch
from MAETF.simulator import MultiAgentEnv
import os
from train_simulation_reward import train_test_split, preprocess_data, load_data

params = get_params()
worker_device = torch.device("cpu") # torch.device("cuda:0")
from_dataset = False


def eval_rnet(net, alloc_batch, targets, env_vect):
    batch_size = env_vect.shape[0]
    loss_func = torch.nn.MSELoss()

    # convert to onehot for further processing
    env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

    # convert numpy array to tensor in shape of input size
    env_vect = numpy_to_input_batch(env_onehot, batch_size, worker_device)
    alloc_batch = numpy_to_input_batch(alloc_batch, batch_size, worker_device)
    targets = numpy_to_input_batch(targets, batch_size, worker_device)

    prediction = net(alloc_batch, env_vect)
    loss = loss_func(prediction, targets)
    print(alloc_batch[:10])
    print(env_vect[:10])
    print(prediction[:10])
    print(targets[:10])
    print(prediction[:4].sum())


if __name__ == '__main__':
    # 3*3
    net = RewardNet(params['n_agent_types'],
                    env_length=params['n_env_types'],
                    norm=params['reward_norm'],
                    n_hidden_layers=5,
                    hidden_layer_size=256)

    # # environment for getting hand-crafted rewards
    # env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
    #                     n_num_agents=params['n_agent_types'],
    #                     n_env_types=params['n_env_types'])

    out_dir = "./logs/reward_logs/reward_weight"
    # params['regress_net_loc']
    # out_dir += '%s_nsamp:%d' % (params['data_method'], params['n_samples'])
    net.load_state_dict(torch.load(out_dir, map_location=worker_device))
    net.eval()

    if from_dataset:
        allocs, ergs, env_type = load_data(params)
        ergs, allocs, env_type = preprocess_data(ergs, allocs, env_type)

        train_allocs, test_allocs = train_test_split(allocs, 0.8)
        train_erg, test_erg = train_test_split(ergs, 0.8)
        train_env, test_env = train_test_split(env_type, 0.8)

        batch_size = 128
        for first in range(0, test_erg.shape[0], batch_size):
            alloc_batch = test_allocs[first:first + batch_size]
            targets = test_erg[first:first + batch_size]
            env_vect = test_env[first:first + batch_size]
            eval_rnet(net, alloc_batch, targets, env_vect)

    else:
        alloc_batch = np.array([0, 0, 8] +
                               [16, 3, 1] +
                               [0, 0, 10] +
                               [4, 17, 1] +
                               [0, 0, 0] +
                               [0, 0, 0] +
                               [20, 20, 20] +
                               [8, 5, 3])
        # env_vect = np.asarray([3] * 8)
        env_vect = np.asarray([0, 1, 2, 3]*2)
        targets = np.zeros_like(env_vect)  # we don't have ground truth for customized data
        eval_rnet(net, alloc_batch, targets, env_vect)
