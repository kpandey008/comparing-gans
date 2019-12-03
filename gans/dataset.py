from torch.utils.data import DataLoader


def get_dataset(choice, transforms=None):
    if choice == 'mnist':
        dataset = MNIST(root, download=True, transform=transforms)
        loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    elif choice == 'fmnist':
        dataset = FashionMNIST(root, download=True, transform=transforms)
        loader = DataLoader(fashion_mnist_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    else:
        dataset = KMNIST(root, download=True, transform=transforms)
        loader = DataLoader(kmnist_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    return dataset, loader
        