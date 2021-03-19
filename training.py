import torch.nn as nn
import torch
from utils import get_params, fake_dataset_generated
from Generator import allocation_generator
from Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
import time
from env import MultiagentEnv
import numpy as np

def train(
    env,
    batch_size: int = 128,
    training_steps: int = 500000,
    learning_rate: float = 0.001,
    print_output_every_n_steps: int = 5000,
    n_critic: int = 5,
):

    params = get_params()
    worker_device = torch.device("cuda:0")
    # Models
    generator = allocation_generator(
        params['n_agent_types'],
        params['env_num'],
        params['env_vect_size'],
        layer_size=params['design_layer_size'],
        design_input_len=params['design_input_len']).to(worker_device)
    discriminator = Discriminator(params['env_num']*params['n_agent_types']).to(worker_device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate
    )

    # loss
    # We don't need this for wgan
    # loss = nn.BCELoss().to(worker_device)

    start_time = time.time()
    writer = SummaryWriter(log_dir="./gan_logs/t:" + str(start_time) + "/")
    for i in range(training_steps):
        batch_size = batch_size

        #if we use an idealized dataset
        true_data = fake_dataset_generated(env)

        # Create noisy input for generator
        # Need float type instead of int
        # noise = torch.rand((batch_size, params['design_input_len']), device=worker_device)
        noise = torch.normal(0, 1, size=(batch_size, params['design_input_len']), device=worker_device)
        env_onehot = torch.zeros([batch_size, params['env_vect_size']], dtype=torch.float, device=worker_device)

        generated_data_raw = generator(noise, env_onehot)
        generated_data = generated_data_raw.reshape(batch_size, -1)
        true_data = torch.tensor(true_data).float().to(worker_device)

        # if i % print_output_every_n_steps:
        #     print("generated data:")
        #     print(generated_data[:10].reshape(-1, 3, 3).argmax(dim=-1))
        #     print("true data: ")
        #     print(true_data[:10].reshape(-1,3,3).argmax(dim=-1))

        # TODO: incorporate batch normalization into the scheme

        if i % n_critic == 0:
            # zero the gradients on each iteration
            generator_optimizer.zero_grad()

            # Train the generator
            # We invert the labels here and don't train the discriminator because we want the generator
            # to make things the discriminator classifies as true.
            generator_discriminator_out = discriminator(generated_data, env_onehot)

            # generator_loss = loss(generator_discriminator_out, true_labels)
            generator_loss = - torch.mean(generator_discriminator_out)
            generator_loss.backward()
            generator_optimizer.step()

            writer.add_scalar('Train' + '/generator_loss', generator_loss.mean(), i)

        else:
            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            true_discriminator_loss = discriminator(true_data, env_onehot)

            # true_discriminator_loss = loss(true_discriminator_out, true_labels)

            # add .detach() here think about this
            generator_discriminator_loss = discriminator(generated_data.detach(), env_onehot)
            # generator_discriminator_loss = loss(
            #     generator_discriminator_out, torch.zeros(batch_size).to(worker_device)
            # )

            # if i % (print_output_every_n_steps+1):
            #     print("false data predict")
            #     print(generator_discriminator_loss[:10])generated_data
            #     print("true label predict")
            #     print(true_discriminator_loss[:10])

            discriminator_loss = torch.mean(generator_discriminator_loss - true_discriminator_loss)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            writer.add_scalar('Train' + '/discriminator_loss', discriminator_loss.mean(), i)

        if i % print_output_every_n_steps == 0:
            print(generated_data_raw[:5].detach().cpu())
            generated_rewards = [env.getReward(alloc.T) for alloc in generated_data_raw[:10].detach().cpu()]
            # print(generated_rewards)
            print(np.mean(generated_rewards))
            torch.save(generator.state_dict(), "gan_logs/t:" + str(start_time) + "/generator_weight")
            torch.save(discriminator.state_dict(), "gan_logs/t:" + str(start_time) + "/discriminator_weight")

    return generator, discriminator


if __name__ == '__main__':
    env = MultiagentEnv()
    train(env)

