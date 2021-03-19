from collections import defaultdict
import numpy as np
import math

def get_params():
    # params contains constants that are shared over all workers
    params = dict()

    # TODO: anneal temperature over many episodes to a lower number
    params['design_input_len'] = 2  # latent variable
    params['design_lr'] = 1e-5
    # params['inv_lr'] = 1e-5
    # params['design_entropy_weight'] = 15  # for entropy in design distributions
    # params['batch_loss_weight'] = bl_rate  # weight for batch difference loss   #originally 0.05
    params['gen_n_hidden'] = 3  # number of hidden layers in the design and reward networks
    params['design_layer_size'] = (64, 128)

    params['batch_size'] = 128
    params['n_agent_types'] = 3
    params['n_env_types'] = 4  # six limbs max, but force left-right symmetry
    params['env_num'] = 4

    ## load the model weights if desired from a previous run
    params['load_from_file'] = 'None'
    params['folder'] = './logs/'
    params['log_interval'] = 300
    params['grad_clip'] = 2

    # params for terrain conv network
    params['env_vect_size'] = params['env_num'] * params['n_env_types']  # the size of the env vector
    params['kernel_size'] = 3
    params['n_channels'] = 32
    params['in_shape'] = (1, 50, 20)  # the input depth image resolution
    params['terrain_batch_norm'] = True  #whether to apply the terrain 2d batch normalization
    params['terrain_dropout'] = False

    #########################################
    #### params that shouldn't be touched ###
    #########################################
    params['reward_loss_scaling'] = 0.2  # alters scale for the reward estimation loss
    params['terrain_norm_weight'] = 0  # penalty on large terrain conv outputs as regularizer

    params['use_replay_buffer'] = False  # we can either use the replay buffer or do on-policy training
    params['replay_buffer_size'] = 500
    params['temperature'] = 1.0  # lower temperature result in closer match to an actual
    params['n_design_steps_ep'] = 4  # number of design steps and mini batch optimizations per episode.
    params['dsgan_max'] = 5  # max for diversity sensitive loss
    params['replay_buffer_size'] = 200
    return params

# re-weight a distribution of robots based on the reward
# assume non-negative reward
def normalize_robot(allocs, rewards):
    batch_size = 128
    def default():
        return 0
    dict = defaultdict(default)
    total_r = 0

    for ind, robot in enumerate(allocs):
        #Get the index of the robot
        reward = rewards[ind]
        robot_str = robot.tostring()
        dict[robot_str] += reward
        total_r += reward

    robot_list = []
    for key in dict:
        r = dict[key]
        if r != 0:
            num = math.ceil(r / total_r * batch_size)
            cur_robot = np.fromstring(key, dtype=float)
            for _ in range(num):
                # TODO: add noise? How?
                robot_list.append(cur_robot)

    # redo if no robot gets sampled (should be rare)
    if(len(robot_list) == 0):
        return normalize_robot(allocs, rewards)

    #resample it back to batch size
    np.random.shuffle(robot_list)
    robot_list = robot_list[:batch_size]

    return robot_list

def fake_dataset_generated(env):
    n_samples = 10
    allocs, rewards = env.generate_random_dist_and_reward(n_samples)
    rewards -= rewards.min()
    alloc_data = normalize_robot(allocs, rewards)
    return alloc_data
