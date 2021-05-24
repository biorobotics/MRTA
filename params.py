'''
Author: Jiaheng Hu
Training Parameters
'''
import os

def get_params():
    # params contains constants that are shared over all workers
    params = dict()

    # TODO: anneal temperature over many episodes to a lower number
    params['design_input_len'] = 2  # latent variable
    params['design_lr'] = 1e-5
    params['gen_n_hidden'] = 3  # number of hidden layers in the design and reward networks
    params['design_layer_size'] = (64, 128)

    params['batch_size'] = 128
    params['n_agent_types'] = 3
    params['n_env_types'] = 4
    params['env_grid_num'] = 4
    params['env_input_len'] = params['env_grid_num'] * params['n_env_types']  # the size of the env vector

    ## load the model weights if desired from a previous run
    params['load_from_file'] = None
    params['log_interval'] = 300

    # WGAN-GP lambda
    params['gp_lambda'] = 10
    params['gen_norm'] = 'none'
    # params['gen_norm'] = 'bn'
    params['dis_norm'] = 'none'
    # params['dis_norm'] = 'ln'

    # params['data_method'] = 'ga'
    # we need to change the constraint for different terrain type
    # params['data_method'] = 'sample_upper_constraint'
    params['data_method'] = 'sample_upper'
    params['n_samples'] = 50

    # params['vary_env'] = 'random'
    params['vary_env'] = 'static'
    # params['vary_env'] = 'discrete'

    params['sim_env'] = True  # Whether we want to use simulated environment
    params['use_regress_net'] = True

    params['folder'] = './logs'

    params['gan_loc'] = os.path.join(params['folder'], 'gan_logs')
    params['regress_net_loc'] = os.path.join(params['gan_loc'], 'reward_weight')
    params['data_loc'] = os.path.join(params['folder'], 'training_data')
    params['reward_loc'] = os.path.join(params['folder'], 'reward_logs', 'reward_agg_dataset')
    params['test_loc'] = os.path.join(params['folder'], 'test_weights')
    return params