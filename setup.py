import configparser
import os
import sys

from setuptools import find_packages, setup

if sys.version_info[0] != 3 or sys.version_info[1] < 6:
    raise RuntimeError("This package requires Python 3.6+")

requirements = [
    # Put package requirements here
    "torch",
    "torchvision",
    "pillow",
    "numpy",
    "matplotlib",
    "tqdm",
    "click",
]

dependency_links = [
]

test_requirements = [
    # Put package test requirements here
]


dev_requirements = [
    # Put development requirements here
]


def forbid_publish():
    """
    Prevent accidental register or upload to PyPI.
    """
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            sys.exit("Command not allowed: {}".format(command))


# This must be called before setup below
forbid_publish()

setup(
    name='cg',
    description='Compares performance of different GAN architectures',
    version="1.0.0",
    url='https://github.com/kpandey008/vowel-consonant-classification',
    author="Kushagra Pandey",
    author_email='kpandeyce008@gmail.com',
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements
    },
    entry_points={
        'console_scripts': [
            'cg = gans.main:cli',
        ],
    },
    dependency_links=dependency_links,
    include_package_data=True,
)
