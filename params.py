'''
Author: Jiaheng Hu
Training Parameters
'''
import os

def get_params():
    # params contains constants that are shared over all workers
    params = dict()

    params['design_input_len'] = 16  # latent variable, this will also change the network capacity
    params['design_lr'] = 1e-5
    params['gen_n_hidden'] = 3  # number of hidden layers in the design and reward networks
    params['design_layer_size'] = (64, 128)

    params['batch_size'] = 128
    params['n_agent_types'] = 3
    params['n_env_types'] = 4
    params['env_grid_num'] = 4
    params['env_input_len'] = params['env_grid_num'] * params['n_env_types']  # the size of the env vector
    params['alloc_len'] = params['n_agent_types'] * params['env_grid_num']

    ## load the model weights if desired from a previous run
    params['load_from_file'] = None
    params['log_interval'] = 300

    # WGAN-GP lambda
    params['gp_lambda'] = 10
    params['gen_norm'] = 'none'
    # params['gen_norm'] = 'bn'
    params['dis_norm'] = 'none'
    # params['dis_norm'] = 'ln'
    # params['reward_norm'] = 'bn'
    params['reward_norm'] = 'none'

    params['data_method'] = 'ga'
    # params['data_method'] = 'sample_upper_constraint'    # we need to change the constraint for different terrain type
    # params['data_method'] = 'sample_upper'
    params['n_samples'] = 50

    # params['vary_env'] = 'random'
    params['vary_env'] = 'static'
    # params['vary_env'] = 'discrete'
    params['agent_num'] = [20, 20, 20]  # [8, 5, 3]

    params['sim_env'] = True  # Whether we want to use simulated environment, otherwise use toy env
    params['use_regress_net'] = True    # we always use regress net when in sim env, but we can also use regress with toy

    params['folder'] = './logs'

    params['gan_loc'] = os.path.join(params['folder'], 'gan_logs')
    params['regress_net_loc'] = os.path.join(params['gan_loc'], 'reward_weight')
    params['data_loc'] = os.path.join(params['folder'], 'training_data')
    params['reward_loc'] = os.path.join(params['folder'], 'reward_logs', 'reward_agg_dataset')  #root folder for rnet
    params['test_loc'] = os.path.join(params['folder'], 'test_weights')

    # params['reward_scale'] = 'log'
    params['reward_scale'] = 'linear'

    return params