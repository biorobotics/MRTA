from collections import defaultdict
import numpy as np
import math
import torch
import torch.autograd as autograd


# re-weight a distribution of assignment based on the reward
# Only take upper corner
def upper_normalize_agent_assignments(allocs, rewards, batch_size=128):
    # #deduct by min first...
    # reward_min = rewards.min()
    #     rewards -= reward_min

    def default():
        return 0
    dict = defaultdict(default)
    total_r = 0
    reward_baseline = rewards.mean() + (rewards.max() - rewards.mean())/2
    rewards -= reward_baseline

    # each robot should only appear once
    for ind, robot in enumerate(allocs):
        # Get the index of the robot
        reward = rewards[ind]
        robot_str = robot.tostring()
        if reward > 0:
            dict[robot_str] += reward
            total_r += reward

    weighted_r = 0
    robot_list = []
    for key in dict:
        r = dict[key]
        if r != 0:
            num = math.ceil(r / total_r * batch_size)
            cur_robot = np.fromstring(key, dtype=float)
            for _ in range(num):
                # TODO: add noise? How?
                robot_list.append(cur_robot)
            weighted_r += num * r
    weighted_r /= len(robot_list)

    # redo if no robot gets sampled (should be rare)
    if len(robot_list) == 0:
        exit("utils.py line 46")

    # re-sample it back to batch size
    np.random.shuffle(robot_list)
    robot_list = robot_list[:batch_size]

    return robot_list, weighted_r + reward_baseline


# re-weight a distribution of assignment based on the reward
# assume non-negative reward
def normalize_agent_assignments(allocs, rewards):
    def default():
        return 0
    dict = defaultdict(default)

    reward_min = rewards.min()
    rewards -= reward_min
    batch_size = 128
    total_r = 0

    # each robot should only appear once
    for ind, robot in enumerate(allocs):
        #Get the index of the robot
        reward = rewards[ind]
        robot_str = robot.tostring()
        dict[robot_str] += reward
        total_r += reward

    weighted_r = 0
    robot_list = []
    for key in dict:
        r = dict[key]
        if r != 0:
            num = math.ceil(r / total_r * batch_size)
            cur_robot = np.fromstring(key, dtype=float)
            for _ in range(num):
                # TODO: add noise? How?
                robot_list.append(cur_robot)
            weighted_r += num * r
    weighted_r /= len(robot_list)

    # redo if no robot gets sampled (should be rare)
    if(len(robot_list) == 0):
        return normalize_agent_assignments(allocs, rewards)

    # resample it back to batch size
    np.random.shuffle(robot_list)
    robot_list = robot_list[:batch_size]

    return robot_list, weighted_r + reward_min


def generate_true_data(env, n_samples, env_type, data_method='sample'):
    # #should only produce uniform and 0.1, 0.1, 0.5, 0.3
    if data_method == 'sample_upper':
        allocs, rewards = env.generate_random_dist_and_reward(n_samples, env_type, constraint=False)
        avg_random_rewards = rewards.mean()
        alloc_data, avg_reward = upper_normalize_agent_assignments(allocs, rewards)
        return alloc_data, avg_reward, avg_random_rewards
    elif data_method == 'sample_upper_constraint':
        allocs, rewards = env.generate_random_dist_and_reward(n_samples, env_type, constraint=True)
        avg_random_rewards = rewards.mean()
        alloc_data, avg_reward = upper_normalize_agent_assignments(allocs, rewards)
        return alloc_data, avg_reward, avg_random_rewards
    elif data_method == 'test':
        allocs, rewards = env.test_dist(env_type)
    elif data_method == 'sample':
        allocs, rewards = env.generate_random_dist_and_reward(n_samples, env_type)
    else:
        exit("utils.py error line 55")
    avg_random_rewards = rewards.mean()
    alloc_data, avg_reward = normalize_agent_assignments(allocs, rewards)
    return alloc_data, avg_reward, avg_random_rewards

# adopted from "https://github.com/caogang/wgan-gp/blob/master/gan_toy.py"
def calc_gradient_penalty(netD, real_data, fake_data, env_onehot, worker_device):
    BATCH_SIZE = real_data.size()[0]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(worker_device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, env_onehot)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(worker_device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def int_to_onehot(l, n):
    a = np.array(l)
    b = np.zeros((a.size, n))
    b[np.arange(a.size), a] = 1
    return b