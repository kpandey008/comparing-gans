import torch


def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))
