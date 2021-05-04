'''
Author: Jiaheng Hu
Training Parameters
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: change the net structure (How about embeddings?)
# disabled BN of wgan gp, might add other normalization techniques. Original paper suggested layer norm

class Discriminator(nn.Module):
    def __init__(self, alloc_length, env_size, norm='none', n_hidden_layers=3, hidden_layer_size=128):
        super(Discriminator, self).__init__()

        self.input_layer = nn.Linear(int(alloc_length), hidden_layer_size // 2)
        # TODO: does this embedding make sense?
        self.embedding_layer = nn.Linear(env_size, hidden_layer_size//2)

        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            self.hidden_list.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.activation = torch.nn.LeakyReLU(0.01)

        self.norm = norm
        if norm == 'bn':
            # batch normalization
            self.bn_list = torch.nn.ModuleList()
            for i in range(n_hidden_layers+1):
                self.bn_list.append(nn.BatchNorm1d(hidden_layer_size))
        elif norm == 'ln':
            # layer normalization
            self.ln_list = torch.nn.ModuleList()
            for i in range(n_hidden_layers + 1):
                self.ln_list.append(nn.LayerNorm(hidden_layer_size))
        elif norm == 'none':
            pass
        else:
            exit("Discriminator.py: wrong normalization argument")

    def forward(self, robot, env_vect):
        # print(design_latent.shape)
        # print(terrain_conv_output.shape)
        x1 = self.activation(self.input_layer(robot))
        x2 = self.activation(self.embedding_layer(env_vect))

        x = torch.cat((x1, x2), dim=-1)

        # #disabled the first layer of bn
        # if use_bn:
        #     x = self.bn_list[0](x)

        for i in range(self.n_hidden_layers):
            x = self.hidden_list[i](x)
            if self.norm == 'bn':
                x = self.bn_list[i+1](x)
            elif self.norm == 'ln':
                x = self.ln_list[i + 1](x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    dis = Discriminator(9)
    print(dis)
    exit()