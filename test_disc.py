# test the discriminator of bgan

from params import get_params
from Discriminator import Discriminator
import torch
import torch.nn.functional as F
from env import MultiAgentEnv

params = get_params()
d_weight_name = "discriminator_weight"
# terrain_value_list = [0, 0.5, 1.0]
terrain_value_list = [0]

# environment for getting hand-crafted rewards
env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                    n_num_agents=params['n_agent_types'],
                    n_env_types=params['n_env_types'])

worker_device = torch.device("cuda:0")


# worker_device = torch.device("cuda:0")
discriminator = Discriminator(alloc_length=params['env_grid_num'] * params['n_agent_types'],
                                  env_size=params["env_input_len"],
                                  norm=params['dis_norm']).to(worker_device)
discriminator.load_state_dict(torch.load(params['folder'] + d_weight_name))
discriminator.eval()

#TODO: unfinished, here
for terrain_value in terrain_value_list:
    print(terrain_value)
    samples = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 2, 0], [1, 2, 1], [2,2,2], [2,2,0]])
    samples_one_hot = F.one_hot(samples, num_classes=3).type(torch.float)
    samples_one_hot_flat = samples_one_hot.view(-1, 9)
    terrain_conv_output = torch.tensor([1 * terrain_value] * samples.shape[0], dtype=torch.float)
    terrain_conv_output = terrain_conv_output.view(-1, 1)
    # print(discriminator(samples_one_hot_flat))
    fake_logits = discriminator(samples_one_hot_flat, terrain_conv_output)
    print(fake_logits)
    fake_out = torch.sigmoid(fake_logits)
    print(fake_out)
    out = torch.cat((samples.type(torch.float), fake_out), 1)
    print(out)

    # samples = torch.tensor( [[0, 2, 2]] * (128*8))
    # samples_one_hot = F.one_hot(samples, num_classes=3).type(torch.float)
    # samples_one_hot_flat = samples_one_hot.view(8, -1, 9)
    # terrain_conv_output = torch.tensor([1 * terrain_value] * samples.shape[0], dtype=torch.float)
    # terrain_conv_output = terrain_conv_output.view(8, -1, 1)
    # # print(discriminator(samples_one_hot_flat))
    # fake_logits = discriminator(samples_one_hot_flat, terrain_conv_output)
    # print(fake_logits[0])
    # fake_out = torch.sigmoid(fake_logits)
    # print(fake_out[0])
    # print(fake_out.shape)
    # # out = torch.cat((samples.type(torch.float), fake_out), 1)