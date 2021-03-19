'''
bgan discriminator
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_params

# TODO: change the net structure (How about embeddings?)

use_bn = True
params = get_params()
env_size = params["env_vect_size"]

class Discriminator(nn.Module):
    def __init__(self, input_length: int, n_hidden_layers=3):
        super(Discriminator, self).__init__()

        hidden_layer_size = 128
        self.input_layer = nn.Linear(int(input_length), hidden_layer_size//2)
        # TODO: does this embedding make sense?
        self.embedding_layer = nn.Linear(env_size, hidden_layer_size//2)

        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            self.hidden_list.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.activation = torch.nn.LeakyReLU(0.01)

        if use_bn:
            # batch normalization
            self.bn_list = torch.nn.ModuleList()
            for i in range(n_hidden_layers+1):
                self.bn_list.append(nn.BatchNorm1d(hidden_layer_size))

    def forward(self, robot, terrain_conv_output):
        # print(design_latent.shape)
        # print(terrain_conv_output.shape)
        x1 = self.activation(self.input_layer(robot))
        x2 = self.activation(self.embedding_layer(terrain_conv_output))

        x = torch.cat((x1, x2), dim=-1)

        ##test disable the first layer of bn
        # if use_bn:
        #     x = self.bn_list[0](x)

        for i in range(self.n_hidden_layers):
            x = self.hidden_list[i](x)
            if use_bn:
                x = self.bn_list[i+1](x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    dis = Discriminator(9)

    print(dis)
    exit()