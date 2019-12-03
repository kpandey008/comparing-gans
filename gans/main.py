import click

from gans.train import train_group
from gans.test import test_group


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
SOURCES = [train_group, test_group]
cli = click.version_option()(
    click.CommandCollection(sources=SOURCES, context_settings=CONTEXT_SETTINGS)
)

if __name__ == '__main__':
    cli()
