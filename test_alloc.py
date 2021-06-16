'''
Author: Jiaheng Hu
Test the trained network
'''

import torch
from utils import generate_true_data, calc_gradient_penalty, int_to_onehot, calc_reward_from_rnet
from params import get_params
from Networks.Generator import AllocationGenerator
from Networks.Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
from MAETF.simulator import MultiAgentEnv
import numpy as np
import os
from collections import defaultdict
import torch.nn.functional as F
from Networks.RewardNet import RewardNet

# n_type_agents = 3
# n_num_grids = 4
# agent_num = [3, 8, 5]
# # turn continuous alloc into discrete assignment
# def get_integer(alloc):
#     alloc = alloc.T
#     int_alloc = np.zeros_like(alloc)
#     for i in range(n_type_agents):
#         remaining = agent_num[i]
#         for j in range(n_num_grids):
#             if j == n_num_grids - 1:
#                 int_alloc[j][i] = remaining
#             else:
#                 cur_num = round(alloc[j][i]*agent_num[i])
#                 cur_num = np.min([remaining, cur_num])
#                 remaining -= cur_num
#                 int_alloc[j][i] = cur_num
#     return int_alloc.T

def test():
    params = get_params()

    batch_size = params['batch_size']
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'],
                        agent_num=params['agent_num'])

    worker_device = torch.device("cuda:0")

    # Models
    generator = AllocationGenerator(
        n_agent=params['n_agent_types'],
        n_env_grid=params['env_grid_num'],
        env_input_len=params['env_input_len'],
        design_input_len=params['design_input_len'],
        norm=params['gen_norm'],
        layer_size=params['design_layer_size']).to(worker_device)

    generator.load_state_dict(torch.load(os.path.join(params['test_loc'], "generator_weight")))
    generator.eval()

    reward_net = RewardNet(params['n_agent_types'],
                           env_length=params['n_env_types'],
                           norm=params['reward_norm'],
                           n_hidden_layers=5, hidden_layer_size=256).to(worker_device)
    reward_net.load_state_dict(torch.load(params['regress_net_loc']))
    reward_net.eval()

    sample_size = 1000
    env_type = [0, 1, 2, 3]
    # env_type = [1, 2, 3, 0]
    # [2, 3, 0, 3]
    env_onehot = torch.tensor(int_to_onehot(env_type, params['n_env_types']),
                              dtype=torch.float, device=worker_device)
    env_onehot = env_onehot.reshape(1, -1).repeat(sample_size, 1)
    noise = torch.normal(0, 1, size=(sample_size, params['design_input_len']), device=worker_device)

    import time
    start_time = time.time()
    generated_data_logits = generator(noise, env_onehot)
    generated_data_raw = F.softmax(generated_data_logits, dim=-1)
    generated_data_raw = generated_data_raw.detach().cpu().numpy().astype(float)
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
    print(f"env type is: {env_type}")


    # int_alloc = [env.get_integer(alloc) for alloc in generated_data_raw[:5]]
    int_alloc = np.array([env.get_integer(alloc) for alloc in generated_data_raw])
    rewards = calc_reward_from_rnet(env, reward_net, int_alloc, env_onehot, sample_size)

    sorted, indices = torch.sort(rewards, descending=True)
    for i in range(5):
        print(sorted[i])
        print(int_alloc[indices[i]])
    # print(rewards.max())
    # print(int_alloc[rewards.argmax()])
    # for i in range(5):
    #     print(int_alloc[i])
    #     print(rewards[i])


    # generated_rewards = np.array([env.get_reward(alloc, env_type) for alloc in generated_data_raw])
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    # print(f"fake data average reward: {np.mean(generated_rewards)}")
    # print(f"fake data max reward: {np.max(generated_rewards)}")
    # print(f"fake data min reward: {np.min(generated_rewards)}")
    # print(generated_rewards)
    # # print(generated_data_raw[np.where(generated_rewards == np.max(generated_rewards))])
    # int_alloc = [env.get_integer(alloc) for alloc in generated_data_raw]
    # # for alloc in int_alloc:
    # #     print(alloc)
    # int_alloc = np.array(int_alloc)
    # # print(int_alloc[0])
    # # print(int_alloc)
    # # hmm this is not right
    # print(np.abs(int_alloc - int_alloc[0]).max(axis=0))
    # print(np.argmax(np.abs(int_alloc - int_alloc[0]), axis=0))
    # # print(np.abs(int_alloc - int_alloc[0]).sum(axis=(1, 2)).mean())
    # # TODO: debug this later
    # # get_count_dict(generated_data_raw, env, env_type)

def get_count_dict(generated_data_raw, env, env_type):
    int_alloc = [env.get_integer(alloc) for alloc in generated_data_raw]
    def default():
        return 0
    dict = defaultdict(default)
    for alloc in int_alloc:
        assignment_str = alloc.tostring()
        dict[assignment_str] += 1
    sorted_key = sorted(dict, key=dict.get)
    for i in range(10):
        max_assign = np.fromstring(sorted_key[-i], dtype=float)
        print(max_assign.reshape((3,4)))

    # TODO: figure out why we are having weird behaviors: max_assign greater than max generated reward...
    # print(dict[sorted_key[-1]])
    # print("reward")
    # print(env.getReward(max_assign.reshape((3,4)).T, env_type))
    # print(int_alloc.index(max_assign.reshape((3,4))))

if __name__ == '__main__':
    test()