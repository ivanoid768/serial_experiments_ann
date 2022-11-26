from typing import List

import numpy as np
from numpy import ndarray

from datapoint_generation import get_batch


def h_f(x: ndarray):
    return 1 / (1 + np.exp(-x))


def h_df(x_e: ndarray):
    return (np.exp(-x_e)) / (1 + np.exp(-x_e)) ** 2


def o_f(x: ndarray):
    return 1 / (1 + np.exp(-x))


def o_df(x_e: ndarray):
    return (np.exp(-x_e)) / (1 + np.exp(-x_e)) ** 2


def mse(tar: ndarray, out: ndarray):
    return (tar - out) ** 2 / out.size


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

    def train(self, batch: List[(ndarray, ndarray)], epoch_cnt: int = 100, lr0: float = 0.01):
        for epoch_idx in range(epoch_cnt):
            lr = (epoch_cnt - epoch_idx) * lr0

            dw_ih: ndarray = np.zeros(self.w_ih.shape)
            dw_ho: ndarray = np.zeros(self.w_ho.shape)

            for inp, target_out in batch:
                out = self.infer_one(inp)

                o_e = (out - target_out) * o_df(out)
                dw_ho += np.dot(o_e[np.newaxis].T, self.l_h[np.newaxis])

                h_e = np.dot(o_e, self.w_ho.T) * h_df(o_e)
                dw_ih += np.dot(h_e[np.newaxis].T, self.l_inp[np.newaxis])

            self.w_ih -= lr * (dw_ih / len(batch))
            self.w_ho -= lr * (dw_ho / len(batch))

    def test(self, batch: List[(ndarray, ndarray)]):
        error = 0

        for inp, target_out in batch:
            out = self.infer_one(inp)
            error += mse(target_out, out)

        error /= len(batch)
        print(f'{error=}')

        return error


if __name__ == '__main__':
    mlp = MLP()
    batch = get_batch(ns_clstr=[2, 2], cluster_std=0.04, n_features=mlp.l_inp.size)
    mlp.test(batch)

    mlp.train(batch=batch, epoch_cnt=100, lr0=0.01)
    mlp.test(batch)

    print(mlp.l_o)
