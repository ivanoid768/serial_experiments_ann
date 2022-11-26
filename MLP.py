from typing import List, Any

import numpy as np
from numpy import ndarray
from scipy.special import expit

from datapoint_generation import get_batch


def sigmoid(x: ndarray):
    # return 1 / (1 + np.exp(-x))
    # return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return expit(x)


def sigmoid_d(x_e: ndarray):
    return (np.exp(-x_e)) / (1 + np.exp(-x_e)) ** 2


def swish(x: ndarray):
    return x * sigmoid(x)


def swish_d(x: ndarray):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))


def h_f(x: ndarray):
    return sigmoid(x)


def h_df(x_e: ndarray):
    return sigmoid_d(x_e)


def o_f(x: ndarray):
    return sigmoid(x)


def o_df(x_e: ndarray):
    return sigmoid_d(x_e)


def mse(tar: ndarray, out: ndarray):
    return np.sum((tar - out) ** 2) / out.size


class MLP:
    def __init__(self):
        self.l_inp = np.zeros(128)
        self.l_h = np.zeros(256)
        self.l_o = np.zeros(2)

        self.w_ih: ndarray = np.random.rand(self.l_inp.size, self.l_h.size) * 0.001
        self.w_ho: ndarray = np.random.rand(self.l_h.size, self.l_o.size) * 0.001

    def infer_one(self, inp: ndarray):
        self.l_inp = np.copy(inp)

        self.l_h = h_f(np.dot(self.l_inp, self.w_ih))
        self.l_o = o_f(np.dot(self.l_h, self.w_ho))

        return self.l_o

    def train(self, batch: List[Any], epoch_cnt: int = 100, lr0: float = 0.01):
        for epoch_idx in range(epoch_cnt):
            lr = (epoch_cnt - epoch_idx) * lr0

            dw_ih: ndarray = np.zeros(self.w_ih.shape)
            dw_ho: ndarray = np.zeros(self.w_ho.shape)

            for inp, target_out in batch:
                out = self.infer_one(inp)

                o_e = (out - target_out) * o_df(self.l_o)
                dw_ho += np.dot(self.l_h[np.newaxis].T, o_e[np.newaxis])

                h_e = np.dot(o_e, self.w_ho.T) * h_df(self.l_h)
                dw_ih += np.dot(self.l_inp[np.newaxis].T, h_e[np.newaxis])

            self.w_ih -= lr * (dw_ih / len(batch))
            self.w_ho -= lr * (dw_ho / len(batch))

    def wta_train(self):
        pass

    def test(self, batch: List[Any]):
        error = 0

        for inp, target_out in batch:
            out = self.infer_one(inp)
            error += mse(target_out, out)

        error /= len(batch)
        print(f'mse: {error}')

        return error


if __name__ == '__main__':
    mlp = MLP()
    batch = get_batch(ns_clstr=[2, 2], cluster_std=0.04, n_features=mlp.l_inp.size)
    mse_start = mlp.test(batch)

    mlp.train(batch=batch, epoch_cnt=100, lr0=0.04)
    mse_trained = mlp.test(batch)

    print(f'{mse_start / mse_trained if mse_trained > 0 else 0}')
