from torch import nn
import numpy as np


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    pass


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    pass
