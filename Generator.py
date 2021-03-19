import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_bn = True
class allocation_generator(nn.Module):

    def __init__(self, n_agent, n_env, env_vect_size,
                 design_input_len,
                 layer_size=(64, 128)):
        super(allocation_generator, self).__init__()

        n_hidden_layers = len(layer_size)
        self.design_input_len = design_input_len
        self.n_env = n_env
        self.n_agent = n_agent

        # TODO: embeddings?
        self.env_vect_size = env_vect_size
        self.input_layer = nn.Linear(n_env + design_input_len, layer_size[0])  # input layer
        self.env_out_layer = nn.Linear(env_vect_size, n_env)
        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers - 1):
            self.hidden_list.append(nn.Linear(layer_size[i], layer_size[i + 1]))
        self.output_layer = nn.Linear(layer_size[i + 1], n_env * n_agent)  # output layer

        # batch normalization
        self.bn_list = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.bn_list.append(nn.BatchNorm1d(layer_size[i]))
        self.env_bn = nn.BatchNorm1d(n_env)
        self.activation = torch.nn.LeakyReLU(0.01)

    # given x a terrain input & noise,
    # return probability distribution over module selections
    def forward(self, design_latent, env_vector):
        # print(design_latent.shape)
        # print(terrain_conv_output.shape)
        x_env = self.env_out_layer(env_vector)
        if use_bn:
            x_env = self.env_bn(x_env)
        x = torch.cat((design_latent, x_env), dim=-1)
        # print(x.shape)
        x = self.activation(self.input_layer(x))
        if use_bn:
            x = self.bn_list[0](x)
        for i in range(self.n_hidden_layers - 1):
            x = self.activation(self.hidden_list[i](x))
            if use_bn:
                x = self.bn_list[i+1](x)
        x = self.output_layer(x)
        x = x.view(-1, self.n_agent, self.n_env)  # add the batch dimension

        # # for bgan, the output is logit, no need for the softmax
        x = F.softmax(x, dim=-1)
        return x

