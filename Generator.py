import torch
import torch.nn as nn

class AllocationGenerator(nn.Module):
    def __init__(self, n_agent, n_env_grid, env_input_len,
                 design_input_len,
                 layer_size=(64, 128), norm='bn'):
        super(AllocationGenerator, self).__init__()

        n_hidden_layers = len(layer_size)
        self.design_input_len = design_input_len
        self.n_agent = n_agent
        self.n_env_grid = n_env_grid
        # preprocess env: map to same length as design input
        self.env_out_layer = nn.Linear(env_input_len, design_input_len)
        # first layer
        self.input_layer = nn.Linear(design_input_len + design_input_len, layer_size[0])
        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers - 1):
            self.hidden_list.append(nn.Linear(layer_size[i], layer_size[i + 1]))
        self.output_layer = nn.Linear(layer_size[i + 1], n_env_grid * n_agent)  # output layer
        self.activation = torch.nn.LeakyReLU(0.01)

        if norm == 'bn':
            self.use_bn = True
            self.env_bn = nn.BatchNorm1d(design_input_len)
            # batch normalization
            self.bn_list = torch.nn.ModuleList()
            for i in range(n_hidden_layers):
                self.bn_list.append(nn.BatchNorm1d(layer_size[i]))
        elif norm == 'none':
            self.use_bn = False
        else:
            exit("Generator.py: wrong normalization argument")



    def forward(self, design_latent, env_vector):
        use_bn = self.use_bn

        x_env = self.env_out_layer(env_vector)
        if use_bn:
            x_env = self.env_bn(x_env)
        x = torch.cat((design_latent, x_env), dim=-1)
        x = self.activation(self.input_layer(x))
        if use_bn:
            x = self.bn_list[0](x)
        for i in range(self.n_hidden_layers - 1):
            x = self.activation(self.hidden_list[i](x))
            if use_bn:
                x = self.bn_list[i+1](x)
        x = self.output_layer(x)
        x = x.view(-1, self.n_agent, self.n_env_grid)  # add the batch dimension

        # this softmax operation is moved to outer loop
        # x = F.softmax(x, dim=-1)

        return x

