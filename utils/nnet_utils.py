import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import re
from torch import Tensor


def states_nnet_to_pytorch_input(states_nnet: np.ndarray, device) -> Tensor:
    states_nnet_tensor = torch.tensor(states_nnet, device=device)

    return states_nnet_tensor


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    pass


# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet
