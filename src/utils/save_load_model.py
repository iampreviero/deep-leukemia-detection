import torch as th
from model.autoencoder import AutoEncoder


def save_parameters(model, directory):
    th.save(model.state_dict(), directory)


def load_model(directory):
    model = AutoEncoder()
    model.load_state_dict(directory)
    return model
