'''
Author: Jiaheng Hu
Reward Network
'''

import torch
import torch.nn as nn
# from params import get_params
#
# params = get_params()
# env_size = params["env_input_len"]
# block_env_size = params['n_env_types']

class RewardNet(nn.Module):
    def __init__(self, input_length, env_length, norm, n_hidden_layers=2, hidden_layer_size=128):
        super(RewardNet, self).__init__()

        # preprocessing - can be changed
        self.input_layer = nn.Linear(int(input_length), hidden_layer_size//2)
        self.embedding_layer = nn.Linear(int(env_length), hidden_layer_size//2)

        # TODO: change to multiply?

        # hidden layer
        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            self.hidden_list.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.activation = torch.nn.ReLU()

        self.drops = nn.Dropout(0.3)
        self.norm = norm
        if norm == 'bn':
            # batch normalization
            self.bn_conv_in = nn.BatchNorm1d(env_length)
            self.bn_list = torch.nn.ModuleList()
            for i in range(n_hidden_layers + 1):
                self.bn_list.append(nn.BatchNorm1d(hidden_layer_size))

    def forward(self, robot, env_vector):
        # print(design_latent.shape)
        # print(terrain_conv_output.shape)
        x1 = self.activation(self.input_layer(robot))
        if self.norm == 'bn':
            env_vector = self.bn_conv_in(env_vector)
        x2 = self.activation(self.embedding_layer(env_vector))

        x = torch.cat((x1, x2), dim=-1)

        for i in range(self.n_hidden_layers):
            x = self.activation(self.hidden_list[i](x))
            # # we don't apply dropout before the last layer
            # if i < self.n_hidden_layers - 1:
            x = self.drops(x)
            if self.norm == 'bn':
                x = self.bn_list[i](x)

        x = self.output_layer(x)
        return x
