import numpy as np
import torchvision as thv
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

np.random.seed(0)


def load_images(data_path, train_split=0.3, rescaling=200, shuffle_dataset=True, batch_size=128, num_workers=4):
    """
    :param data_path:
    :param train_split:
    :param rescaling:
    :param shuffle_dataset:
    :param batch_size:
    :param num_workers:
    :return:
    """
    dataset = thv.datasets.ImageFolder(
        root=data_path,
        transform=Compose([Resize(rescaling), ToTensor()])
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                   sampler=valid_sampler)

    return train_loader, validation_loader
