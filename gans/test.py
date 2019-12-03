import os

import click
import torch

from gans.models.generator import Generator
from gans.util import display_grid


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@click.group()
def test_group():
    pass


@test_group.command()
@click.argument('gen_path', help='Generator model path')
@click.argument('code_size', help='The code size of the code input to the generator')
@click.argument('num_samples', help='Number of Samples to generate')
def generate_samples(gen_path, code_size, num_samples):
    """Generates samples from the generator
    """
    if not os.path.isfile(gen_path):
        raise ValueError(f'No model exists in the path: {gen_path}')
    
    # Create a generator
    gen_model = Generator(code_size)
    gen_model.load_state_dict(torch.load(gen_model_name))
    gen_model.to(device)

    # Generate the samples
    noise_batch = torch.randn(num_samples, code_size).to(device)
    gen_model.eval()
    generated_samples = gen_model(noise_batch)

    # Visualize the outputs
    _display_grid(gen_sample_outputs, num_rows=8)


if __name__ == "__main__":
    test_group()
