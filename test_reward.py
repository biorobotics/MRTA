# Author: Jiaheng Hu
# Sample a batch of env and alloc, test the reward net prediction

import numpy as np
from RewardNet import RewardNet
import torch
from params import get_params
from utils import int_to_onehot
from env import MultiAgentEnv
import os

params = get_params()

worker_device = torch.device("cuda:0")

if __name__ == '__main__':
    # 3*3
    net = RewardNet(params['env_grid_num'] * params['n_agent_types'], n_hidden_layers=5, hidden_layer_size=256).to(worker_device)
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])



    loss_func = torch.nn.MSELoss()

    out_dir = os.path.join('reward_logs/', 'reward_agg')
    # out_dir += '%s_nsamp:%d' % (params['data_method'], params['n_samples'])
    net.load_state_dict(torch.load(os.path.join(out_dir, "reward_weight")))
    net.eval()

    test_iter = 1
    sample_size = 10
    for i in range(test_iter):
        # env_vect = np.random.choice(4, (sample_size, 4))
        env_vect = np.array([[0,1,2,3]]*sample_size)
        # print(env_vect)
        # generate_random_dist_and_reward_per_env
        alloc_batch, rewards = env.generate_fixed_dist_and_reward_per_env(env_vect)

        # convert to onehot for further processing
        env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

        env_onehot = torch.from_numpy(env_onehot).view(sample_size, -1).float().to(worker_device)
        # convert numpy array to tensor in shape of input size
        alloc_batch = torch.from_numpy(alloc_batch).view(sample_size, -1).float().to(worker_device)
        rewards = torch.from_numpy(rewards).float().reshape((-1, 1)).to(worker_device)

        prediction = net(alloc_batch, env_onehot)
        loss = loss_func(prediction, rewards)
        print(env_vect)
        print(alloc_batch)
        print(rewards)
        print(prediction)
        print(loss)

    # env_vect = [0, 1, 2, 3]
    # alloc_batch = np.array([[1, 0, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])
    # # convert to onehot for further processing
    # env_onehot = np.array(int_to_onehot(env_vect, params['n_env_types']))
    #
    # env_onehot = torch.from_numpy(env_onehot).view(1, -1).float().to(worker_device)
    # # convert numpy array to tensor in shape of input size
    # alloc_batch = torch.from_numpy(alloc_batch).view(1, -1).float().to(worker_device)
    #
    # prediction = net(alloc_batch, env_onehot)
    # print(prediction)