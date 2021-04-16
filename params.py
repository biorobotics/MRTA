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
    params['folder'] = './logs/'
    params['log_interval'] = 300

    # WGAN-GP lambda
    params['gp_lambda'] = 10
    params['gen_norm'] = 'bn'
    params['dis_norm'] = 'none'
    params['n_samples'] = 50

    params['data_method'] = 'sample_upper_constraint'

    params['vary_env'] = 'static'
    return params