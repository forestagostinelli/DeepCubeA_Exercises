from typing import List, Tuple, Any
from environments.environment_abstract import Environment, State
from utils.nnet_utils import states_nnet_to_pytorch_input
import numpy as np
import math


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[Any], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])

        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def evaluate_cost_to_go(nnet, device, env: Environment, states: List[State], outputs: np.array):
    for cost_to_go in np.unique(outputs):
        idxs_targ: np.array = np.where(outputs == cost_to_go)[0]
        states_targ: List[State] = [states[idx] for idx in idxs_targ]
        states_targ_nnet: np.ndarray = env.state_to_nnet_input(states_targ)

        out_nnet = nnet(states_nnet_to_pytorch_input(states_targ_nnet, device).float()).cpu().data.numpy()

        mse = float(np.mean((out_nnet - cost_to_go) ** 2))
        print("Cost-To-Go: %i, Ave DNN Output: %f, MSE: %f" % (cost_to_go, float(np.mean(out_nnet)), mse))

