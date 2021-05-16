from collections import defaultdict
import numpy as np
import math
import torch
import torch.autograd as autograd
from evo_3 import evolve_one_gen
from scipy.special import softmax
from params import get_params
# helper functions have access to params
params = get_params()


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


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# convert env to form ready to be taken by neural nets
def env_to_n_onehot(env_type, n_samples):
    env_vect = np.array([env_type] * n_samples)
    # convert to onehot for further processing
    env_onehot = np.array([int_to_onehot(vect, params['n_env_types']) for vect in env_vect])
    # env_onehot = torch.from_numpy(env_onehot).view(n_samples, -1).float().to(worker_device)
    return env_onehot

def numpy_to_input_batch(array, n_samples):
    vect = torch.from_numpy(array).view(n_samples, -1).float().cuda()
    return vect

def convert_erg_to_reward(ergs):
    ergs = torch.clip(ergs, 5, 14)
    rewards = 14 - ergs
    return rewards


def generate_true_regress_data(env, n_samples, env_type, net, data_method='sample', fake_data=None):
    # we only need the dist, replace this later
    allocs, _ = env.generate_random_dist_and_reward(n_samples, env_type, constraint=False)
    if data_method is not 'sample_upper':
        exit("rest of the sampling method needs to be double checked, utils.py")

    envs = env_to_n_onehot(env_type, n_samples)
    envs_torch = numpy_to_input_batch(envs, n_samples)
    allocs_torch = numpy_to_input_batch(allocs, n_samples)
    ergs = net(allocs_torch, envs_torch)
    # process the reward, since it is not really meaningful rn
    rewards = convert_erg_to_reward(ergs)
    alloc_data, avg_reward = resample_data(allocs, rewards, data_method)
    avg_random_rewards = rewards.mean()
    return alloc_data, avg_reward, avg_random_rewards

def resample_data(allocs, rewards, data_method):
    if data_method == 'sample_upper':
        alloc_data, avg_reward = upper_normalize_agent_assignments(allocs, rewards)
    elif data_method == 'sample_upper_constraint':
        alloc_data, avg_reward = upper_normalize_agent_assignments(allocs, rewards)
    elif data_method == 'test':
        alloc_data, avg_reward = normalize_agent_assignments(allocs, rewards)
    elif data_method == 'sample':
        alloc_data, avg_reward = normalize_agent_assignments(allocs, rewards)
    return alloc_data, avg_reward

def generate_true_data(env, n_samples, env_type, data_method='sample', fake_data=None):
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
        # #should only produce uniform and 0.1, 0.1, 0.5, 0.3
        allocs, rewards = env.test_dist(env_type)
    elif data_method == 'sample':
        allocs, rewards = env.generate_random_dist_and_reward(n_samples, env_type)
    elif data_method == 'ga':
        # first, obtain the generated data
        if fake_data is None:
            exit("utils.py line 128 fatal error")
        else:
            #generate the next generation of population
            #take the current population, and evolve it
            allocs = np.array(fake_data.detach().cpu())
            fitness = np.array([env.get_reward(alloc, env_type) for alloc in softmax(allocs, axis=-1)])
            new_data = evolve_one_gen(allocs.reshape(128, 12), fitness)
            new_data = softmax(new_data.reshape(128, 3, 4), axis=-1)

            # TODO: what if we return logits as well, and discriminator also takes in logits?
            new_fit_avg = np.mean([env.get_reward(alloc, env_type) for alloc in new_data])
            return new_data.reshape(128, 12), new_fit_avg, fitness.mean()
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