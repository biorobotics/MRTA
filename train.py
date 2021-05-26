'''
Author: Jiaheng Hu
Train the Network
'''

import torch
from utils import generate_true_data, calc_gradient_penalty, int_to_onehot, generate_true_regress_data, calc_reward_from_rnet
from params import get_params
from Networks.Generator import AllocationGenerator
from Networks.Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch.nn.functional as F
from Networks.RewardNet import RewardNet


def train(
    training_steps: int = 500000,
    learning_rate: float = 0.001,
    print_output_every_n_steps: int = 5000,
    n_critic: int = 5,
):

    params = get_params()
    batch_size = params['batch_size']

    if params['sim_env']:
        from MAETF.simulator import MultiAgentEnv
        env_name = 'simenv'
    else:
        from toy_env import MultiAgentEnv
        env_name = 'toyenv'
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])

    worker_device = torch.device("cuda:0")

    # Models
    generator = AllocationGenerator(
        n_agent=params['n_agent_types'],
        n_env_grid=params['env_grid_num'],
        env_input_len=params['env_input_len'],
        design_input_len=params['design_input_len'],
        norm=params['gen_norm'],
        layer_size=params['design_layer_size']).to(worker_device)
    discriminator = Discriminator(alloc_length=params['env_grid_num'] * params['n_agent_types'],
                                  env_size=params["env_input_len"],
                                  norm=params['dis_norm']).to(worker_device)

    if params['load_from_file']:
        # load weight
        exit("err: change the path (line 56 train.py)")
        out_dir = "./gan_logs/"+params['load_from_file']
        generator.load_state_dict(torch.load("./gan_logs/"+params['load_from_file']+"/generator_weight"))
        discriminator.load_state_dict(torch.load("./gan_logs/"+params['load_from_file']+"/discriminator_weight"))
    else:
        out_dir = os.path.join(params['gan_loc'], "")
        out_dir += '%s_nsamp:%d_%s' % (params['data_method'], params['n_samples'], env_name)
        out_dir += '_%s_%s_%s_rnet:%i' % (params['vary_env'],
                                          params['gen_norm'],
                                          params['dis_norm'],
                                          params['use_regress_net'])
        # out_dir += '_%s_gpl:%g' % (params['dis_norm'], params['gp_lambda'])
        # out_dir += '_atypes:%d_enum:%d_etypes:%d' % (
        #     params['n_agent_types'], params['env_grid_num'], params['n_env_types'])

        if os.path.exists(out_dir):
            cmd = 'rm %s/*' % out_dir
            os.system(cmd)

    if params['use_regress_net']:
        reward_net = RewardNet(params['n_agent_types'],
                               env_length=params['n_env_types'],
                               n_hidden_layers=5, hidden_layer_size=256).to(worker_device)
        reward_net.load_state_dict(torch.load(params['regress_net_loc']))
        reward_net.eval()

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate
    )

    writer = SummaryWriter(log_dir=out_dir)
    for i in range(training_steps):
        # TODO: env type should be the actual env type: env_type pass into onehot
        if params['vary_env'] == 'static':
            env_type = [0, 1, 2, 3]
        elif params['vary_env'] == 'discrete':
            env_type_list = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 3]]
            env_dex = np.random.randint(len(env_type_list))
            env_type = env_type_list[env_dex]
        elif params['vary_env'] == 'random':
            env_type = np.random.choice(4, 4)
        else:
            exit("error train.py line 71")
        env_onehot = torch.tensor(int_to_onehot(env_type, params['n_env_types']),
                                  dtype=torch.float, device=worker_device)
        env_onehot = env_onehot.reshape(1, -1).repeat(batch_size, 1)

        # Create noisy input for generator
        # noise = torch.rand((batch_size, params['design_input_len']), device=worker_device)
        noise = torch.normal(0, 1, size=(batch_size, params['design_input_len']), device=worker_device)

        generated_data_logits = generator(noise, env_onehot)
        generated_data_raw = F.softmax(generated_data_logits, dim=-1)
        generated_data = generated_data_raw.reshape(batch_size, -1)

        # generated random data based on reward net
        if params['use_regress_net']:
            true_data, true_avg_r, true_raw_r = generate_true_regress_data(env, params['n_samples'],
                                                                           env_type, reward_net,
                                                                           data_method=params['data_method'],
                                                                           fake_data=generated_data_logits)
        else:
            # if we use an idealized dataset
            true_data, true_avg_r, true_raw_r = generate_true_data(env, params['n_samples'], env_type,
                                                                   data_method=params['data_method'],
                                                                   fake_data=generated_data_logits)

        true_data = torch.tensor(true_data).float().to(worker_device)

        if i % n_critic == 0:
            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            # Train the generator
            # We invert the labels here and don't train the discriminator because we want the generator
            # to make things the discriminator classifies as true.
            generator_discriminator_out = discriminator(generated_data, env_onehot)

            # generator_loss = loss(generator_discriminator_out, true_labels)
            generator_loss = - torch.mean(generator_discriminator_out)
            generator_loss.backward()
            generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_loss = discriminator(true_data, env_onehot).mean()

        # true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_loss = discriminator(generated_data.detach(), env_onehot).mean()
        gp = calc_gradient_penalty(discriminator, true_data, generated_data.detach(),
                                   env_onehot, worker_device) * params['gp_lambda']
        discriminator_loss_log = generator_discriminator_loss.detach() - true_discriminator_loss.detach()
        discriminator_loss = generator_discriminator_loss - true_discriminator_loss + gp
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # log interval
        if i % (2 * n_critic) == 0:
            writer.add_scalar('Train' + '/generator_loss', generator_loss.mean(), i)
            if params['vary_env'] == 'static' or params['vary_env'] == 'random':
                writer.add_scalar('Disc' + '/generator_discriminator_loss',
                                  generator_discriminator_loss.detach().mean(), i)
                writer.add_scalar('Disc' + '/true_discriminator_loss',
                                  true_discriminator_loss.detach().mean(), i)
                writer.add_scalar('Train' + '/discriminator_loss',
                                  discriminator_loss_log.mean(), i)
            elif params['vary_env'] == 'discrete':
                writer.add_scalar('Disc' + '/generator_discriminator_loss' + str(env_dex),
                                  generator_discriminator_loss.detach().mean(), i)
                writer.add_scalar('Disc' + '/true_discriminator_loss' + str(env_dex),
                                  true_discriminator_loss.detach().mean(), i)
                writer.add_scalar('Train' + '/discriminator_loss' + str(env_dex),
                                  discriminator_loss_log.mean(), i)



        if i % print_output_every_n_steps == 0:
            print(f"env type is: {env_type}")
            int_alloc = np.array([env.get_integer(alloc) for alloc in generated_data_raw.detach().cpu().numpy()])
            for alloc in int_alloc[:5]:
                print(alloc)

            if params['sim_env']:
                generated_rewards = calc_reward_from_rnet(env, reward_net, int_alloc, env_onehot, batch_size)
            else:
                generated_rewards = np.array([env.get_integer_reward(alloc, env_type) for alloc in int_alloc])

            print(f"fake data average reward: {generated_rewards.mean()}")
            writer.add_scalar('R' + '/fake_avg_reward', generated_rewards.mean(), i)

            print(f"real data average reward: {true_avg_r}")
            print(f"random sample average reward: {true_raw_r}")

            writer.add_scalar('R' + '/true_avg_reward', true_avg_r, i)
            writer.add_scalar('R' + '/true_raw_reward', true_raw_r, i)
            torch.save(generator.state_dict(), os.path.join(out_dir, "generator_weight"))
            torch.save(discriminator.state_dict(), os.path.join(out_dir, "discriminator_weight"))
    return generator, discriminator


if __name__ == '__main__':
    train()

