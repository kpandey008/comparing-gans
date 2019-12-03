import os

import click
import torch
import torch.nn as nn
import torch.optim as optim

from gans.models.generator import Generator
from gans.models.discriminator import Discriminator
from gans.util import _get_transforms, _train_one_epoch, _load_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@click.group()
def train_group():
    pass


@train_group.command()
@click.option('--dataset', '-d', default='mnist', click.Choice=(['mnist', 'fmnist', 'kmnist'], case_sensitive=False))
@click.argument('save_dir', help='Directory to save the models in. The provided dir need not exist')
def train(dataset, save_dir):
    """Trains a Generative Adversarial Network
    """
    # load the param config
    config = _load_config('config.yaml')

    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    code_size = config['code_size']
    log_step = config['log_step']
    k = config['k']
    batch_size = config['batch_size']

    # Load the datasets
    T = _get_transforms()
    dataset, loader = get_dataset(dataset, transforms=T)

    # Define the models
    gen_model = Generator(code_size, training=True)
    gen_loss = nn.BCELoss()
    gen_model = gen_model.to(device)
    gen_optim = torch.optim.Adam(gen_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    disc_model = Discriminator()
    disc_loss = nn.BCELoss()
    disc_model = disc_model.to(device)
    disc_optim = torch.optim.Adam(disc_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Main training loop
    gen_loss_prof = []
    disc_loss_prof = []
    for epoch_idx in range(num_epochs):
        gen_loss_profile, disc_loss_profile = _train_one_epoch(disc_model, gen_model,
                                                            disc_loss, gen_loss,
                                                            disc_optim, gen_optim,
                                                            loader, k,
                                                            epoch_idx, log_step=log_step)
        gen_loss_prof.extend(gen_loss_profile)
        disc_loss_prof.extend(disc_loss_profile)

    # Save the models after training
    os.makedirs(save_dir, exist_ok=True)
    torch.save(gen_model.state_dict(), 'generator.pt')
    torch.save(disc_model.state_dict(), 'discriminator.pt')

    print('Models Saved. Done')


@train_group.command()
def sanity_check_model():
    """
    Ensures that the forward pass through the GAN works
    """
    # Sanity check the discriminator and the generator
    pass


if __name__ == "__main__":
    train_group()
