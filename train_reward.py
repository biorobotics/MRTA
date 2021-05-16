# Author: Jiaheng Hu
# Train a reward network through regression

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from RewardNet import RewardNet
import torch
from params import get_params
from utils import int_to_onehot
from env import MultiAgentEnv
import os

# local hyperparams
batch_size = 128
test_size = 500

epoch_before_new_data = 5
total_loop = 50000
buffer_size = 50000
log_interval = 10
new_sample_size = 0  # 50  # if we want to add new data incrementally

params = get_params()

worker_device = torch.device("cuda:0")

if __name__ == '__main__':
    # 3*3
    net = RewardNet(params['env_grid_num'] * params['n_agent_types'], n_hidden_layers=5, hidden_layer_size=256).to(worker_device)
    # environment for getting hand-crafted rewards
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                        n_num_agents=params['n_agent_types'],
                        n_env_types=params['n_env_types'])


    reward_optimizer_params_list = []
    reward_optimizer_params_list += list(net.parameters())

    # Define Optimizer and Loss Function
    optimizer = torch.optim.Adam(reward_optimizer_params_list, lr=0.001)
    loss_func = torch.nn.MSELoss()

    out_dir = os.path.join('reward_logs/', 'reward_agg')
    # out_dir += '%s_nsamp:%d' % (params['data_method'], params['n_samples'])
    if os.path.exists(out_dir):
        cmd = 'rm %s/*' % out_dir
        os.system(cmd)

    writer = SummaryWriter(log_dir=out_dir)

    for i in range(total_loop):
        # train loop
        total_loss = 0
        for _ in range(epoch_before_new_data):
            env_vect = np.random.choice(4, (batch_size, 4))
            alloc_batch, rewards = env.generate_random_dist_and_reward_per_env(env_vect)
            # convert to onehot for further processing
            env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])

            env_vect = torch.from_numpy(env_onehot).view(batch_size, -1).float().to(worker_device)
            # convert numpy array to tensor in shape of input size
            alloc_batch = torch.from_numpy(alloc_batch).view(batch_size, -1).float().to(worker_device)
            rewards = torch.from_numpy(rewards).float().reshape((-1, 1)).to(worker_device)

            prediction = net(alloc_batch, env_vect)
            loss = loss_func(prediction, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= epoch_before_new_data

        if new_sample_size > 0:
            exit("if we are running with real data, online training hasn't been implemented")
            # get new training data
            train_robot, train_terrains, train_rewards = reward_data_generate_conv(batch_size=new_sample_size)

            train_robot_db = np.concatenate([train_robot_db, train_robot])
            train_terrains_db = np.concatenate([train_terrains_db, train_terrains])
            train_rewards_db = np.concatenate([train_rewards_db, train_rewards])

            # check if data size exceed capacity
            cur_len_buf = train_robot_db.shape[0]
            if cur_len_buf > buffer_size:
                train_robot_db = train_robot_db[cur_len_buf - buffer_size:]
                train_terrains_db = train_terrains_db[cur_len_buf - buffer_size:]
                train_rewards_db = train_rewards_db[cur_len_buf - buffer_size:]

        if i % log_interval == 0:
            # # TODO: deal with testing. RN we use train loss directly (since we have infinite data)
            # # get evaluation
            # net.eval()
            # with torch.no_grad():
            #     test_terr_out = terr_conv(test_terrains)
            #     predicted = net(test_robot_batch, test_terr_out)
            #     print(predicted[:10])
            #     print(test_rewards[:10])
            #     eval_loss = loss_func(predicted, test_rewards)
            # net.train()

            # print(prediction)
            # print(rewards)
            # print(rewards.mean())
            print(f"train loss: {total_loss}") #, test loss: {eval_loss}")
            writer.add_scalar('Train' + '/reward_loss', total_loss, i)
            # writer.add_scalar('Test' + '/reward_loss', eval_loss, i)
            torch.save(net.state_dict(), os.path.join(out_dir, "reward_weight"))
            # torch.save(terr_conv.state_dict(), "reward_logs/t:" + str(start_time) + "/terrain_weight")