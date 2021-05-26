# Author: Jiaheng Hu
# Train a reward network through regression
# use pre_collected data

# import sys
# # tem solution, use file from the other directory
# # eventually move everything into this folder
# # scaling problem
# sys.path.insert(1, '/home/jeff/Projects/multiagent_allocation')
#
#

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from Networks.RewardNet import RewardNet
import torch
from params import get_params
from utils import int_to_onehot
from MAETF.simulator import MultiAgentEnv
import os

# TODO: later, change to add env type as part of dataset as well

# local hyperparams
batch_size = 128
test_size = 500

epoch_before_new_data = 5
total_loop = 50000
buffer_size = 50000
log_interval = 1000
new_sample_size = 0  # 50  # if we want to add new data incrementally

params = get_params()

worker_device = torch.device("cuda:0")
env = MultiAgentEnv()

def train_test_split(data, portion):
    num_of_data = data.shape[0]
    n_train = int(num_of_data * portion)
    train = data[:n_train]
    test = data[n_train:]
    return train, test

def load_data(params):
    data_loc = params['data_loc']
    allocs = np.load(os.path.join(data_loc, "assignments.npy"))
    ergs = np.load(os.path.join(data_loc, "ergs.npy")) * 1000  # Just to make it numerically stable
    env_type = np.load(os.path.join(data_loc, "env_labels.npy"))
    assert (allocs.shape[0] == ergs.shape[0] == env_type.shape[0])
    return allocs, ergs, env_type

# change shape, flatten entries, remove nan
def preprocess_data(ergs, allocs, env_type):
    # print(env.get_integer(allocs[0]))
    allocs = np.array([env.get_integer(alloc).T for alloc in allocs]).reshape(-1, env.n_types_agents)
    ergs = ergs.flatten()
    env_type = env_type.flatten()
    idx_to_rm = np.isnan(ergs)
    allocs = allocs[~idx_to_rm]
    ergs = ergs[~idx_to_rm].clip(None, 4)
    env_type = env_type[~idx_to_rm]
    # print(allocs.shape)
    # print(ergs.shape)
    # print(env_type.shape)
    # print(allocs[0])
    return ergs, allocs, env_type



if __name__ == '__main__':
    net = RewardNet(params['n_agent_types'],
                    env_length=params['n_env_types'],
                    n_hidden_layers=5,
                    hidden_layer_size=256).to(worker_device)

    allocs, ergs, env_type = load_data(params)
    ergs, allocs, env_type = preprocess_data(ergs, allocs, env_type)
    # print(np.max(ergs))
    # print(np.min(ergs))
    # import matplotlib.pyplot as plt
    # _ = plt.hist(ergs, bins='auto')  # arguments are passed to np.histogram
    # plt.show()
    # # think about how to convert this to reasonable scale...
    # # problem: can't know number from the vector: the number also depends on how the previous ones are calculated
    # # Solution: calculate number based on the float value, reward network works on numbers instead of portions
    # exit()
    train_allocs, test_allocs = train_test_split(allocs, 0.8)
    train_erg, test_erg = train_test_split(ergs, 0.8)
    train_env, test_env = train_test_split(env_type, 0.8)

    reward_optimizer_params_list = []
    reward_optimizer_params_list += list(net.parameters())

    # Define Optimizer and Loss Function
    optimizer = torch.optim.Adam(reward_optimizer_params_list, lr=0.001)
    loss_func = torch.nn.MSELoss()

    out_dir = params['reward_loc']
    # out_dir += '%s_nsamp:%d' % (params['data_method'], params['n_samples'])
    if os.path.exists(out_dir):
        cmd = 'rm %s/*' % out_dir
        os.system(cmd)

    writer = SummaryWriter(log_dir=out_dir)

    for i in range(total_loop):
        # train loop
        total_loss = 0
        counter = 0
        for first in range(0, train_erg.shape[0], batch_size):

            alloc_batch = train_allocs[first:first + batch_size]
            targets = train_erg[first:first + batch_size]

            cur_batch_size = targets.shape[0]

            # if we are using the whole environment for prediction
            # convert to onehot for further processing
            # env_vect = np.asarray([0, 1, 2, 3] * cur_batch_size)
            # env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

            env_vect = train_env[first:first + batch_size]
            env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

            env_vect = torch.from_numpy(env_onehot).view(cur_batch_size, -1).float().to(worker_device)
            # convert numpy array to tensor in shape of input size
            alloc_batch = torch.from_numpy(alloc_batch).view(cur_batch_size, -1).float().to(worker_device)
            targets = torch.from_numpy(targets).float().reshape((-1, 1)).to(worker_device)

            prediction = net(alloc_batch, env_vect)
            loss = loss_func(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            counter += 1
        total_loss /= counter

        if i % log_interval == 0:
            # TODO: log frequency is not right
            # get evaluation
            net.eval()
            test_loss = 0
            counter = 0
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
                test_loss += loss.item()
                counter += 1
            print(prediction[:10])
            print(targets[:10])
            test_loss /= counter
            net.train()

            # print(prediction)
            # print(rewards)
            # print(rewards.mean())
            print(f"train loss: {total_loss}, test loss: {test_loss}")
            writer.add_scalar('Train' + '/reward_loss', total_loss, i)
            writer.add_scalar('Test' + '/reward_loss', test_loss, i)
            torch.save(net.state_dict(), os.path.join(out_dir, "reward_weight"))