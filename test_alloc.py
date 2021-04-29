import torch
from utils import generate_true_data, calc_gradient_penalty
from params import get_params
from Generator import allocation_generator
from Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
from env import MultiAgentEnv
import numpy as np
import os
from collections import defaultdict
import torch.nn.functional as F

def test():
    params = get_params()

    batch_size = params['batch_size']
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])

    worker_device = torch.device("cuda:0")

    # Models
    generator = allocation_generator(
        n_agent=params['n_agent_types'],
        n_env_grid=params['env_grid_num'],
        env_input_len=params['env_input_len'],
        design_input_len=params['design_input_len'],
        norm=params['gen_norm'],
        layer_size=params['design_layer_size']).to(worker_device)
    discriminator = Discriminator(alloc_length=params['env_grid_num'] * params['n_agent_types'],
                                  env_size=params["env_input_len"],
                                  norm=params['dis_norm']).to(worker_device)

    generator.load_state_dict(torch.load("./test_weights/generator_weight"))
    discriminator.load_state_dict(torch.load("./test_weights/discriminator_weight"))

    # TODO: why would evaluation cause problem? [Oh the batch norm of the generator]
    # generator.eval()
    # discriminator.eval()

    sample_size = 1000
    env_type = [0, 1, 2, 3]
    env_onehot = torch.zeros([sample_size, params['env_input_len']], dtype=torch.float, device=worker_device)
    noise = torch.normal(0, 1, size=(sample_size, params['design_input_len']), device=worker_device)

    generated_data_logits = generator(noise, env_onehot)
    generated_data_raw = F.softmax(generated_data_logits, dim=-1)
    generated_data_raw = generated_data_raw.detach().cpu().numpy().astype(float)
    print(f"env type is: {env_type}")
    int_alloc = [env.getInteger(alloc.T).T for alloc in generated_data_raw[:5]]
    for alloc in int_alloc:
        print(alloc)
    generated_rewards = np.array([env.getReward(alloc.T, env_type) for alloc in generated_data_raw])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print(f"fake data average reward: {np.mean(generated_rewards)}")
    print(f"fake data max reward: {np.max(generated_rewards)}")
    print(f"fake data min reward: {np.min(generated_rewards)}")
    print(generated_rewards)
    print(generated_data_raw[np.where(generated_rewards == np.max(generated_rewards))])
    # TODO: debug this later
    # get_count_dict(generated_data_raw, env, env_type)

def get_count_dict(generated_data_raw, env, env_type):
    int_alloc = [env.getInteger(alloc.T).T for alloc in generated_data_raw]
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