from typing import List
import numpy as np

import torch
from torch import nn

from environments.environment_abstract import Environment, State
from utils import env_utils
from utils.nnet_utils import train_nnet, states_nnet_to_pytorch_input
from utils.misc_utils import split_evenly

import pickle


def sample_training_data(states: List[State], outputs: np.ndarray, env: Environment, num_samp_total: int):
    max_cost_to_go: int = int(np.max(outputs))

    samp_idxs: np.array = np.zeros(0, dtype=np.int)
    num_per_cost_to_go: List[int] = split_evenly(num_samp_total, max_cost_to_go + 1)

    for cost_to_go, num_samp in zip(range(max_cost_to_go + 1), num_per_cost_to_go):
        ctg_idxs = np.where(outputs[:, 0] == cost_to_go)[0]
        ctg_samp_idxs = np.random.choice(ctg_idxs, size=num_samp)

        samp_idxs = np.concatenate((samp_idxs, ctg_samp_idxs))

    np.random.shuffle(samp_idxs)

    states_samp: List[State] = [states[idx] for idx in samp_idxs]
    states_nnet: np.ndarray = env.state_to_nnet_input(states_samp)
    outputs: np.ndarray = outputs[samp_idxs]

    return states_nnet, outputs


def main():
    # get environment
    env: Environment = env_utils.get_environment("puzzle8")

    # get nnet model
    nnet: nn.Module = env.get_nnet_model()
    device = torch.device('cpu')
    batch_size: int = 100
    num_itrs: int = 10000

    # get data
    print("Preparing Data\n")
    data = pickle.load(open("data/data.pkl", "rb"))

    states_nnet, outputs = sample_training_data(data['states'], data['output'], env, batch_size*num_itrs)

    # train with supervised learning
    print("Training DNN\n")
    nnet.train()
    train_nnet(nnet, states_nnet, outputs, batch_size, num_itrs, 0)

    # get performance
    print("Evaluating DNN\n")
    nnet.eval()
    for cost_to_go in np.unique(data["output"]):
        idxs_targ: np.array = np.where(data["output"] == cost_to_go)[0]
        states_targ: List[State] = [data["states"][idx] for idx in idxs_targ]
        states_targ_nnet: np.ndarray = env.state_to_nnet_input(states_targ)

        out_nnet = nnet(states_nnet_to_pytorch_input(states_targ_nnet, device)).cpu().data.numpy()

        mse = float(np.mean((out_nnet - cost_to_go) ** 2))
        print("Cost-To-Go: %i, Ave DNN Output: %f, MSE: %f" % (cost_to_go, float(np.mean(out_nnet)), mse))


if __name__ == "__main__":
    main()
