from typing import List, Any

import numpy as np
from numpy import ndarray
from scipy.special import expit

from datapoint_generation import generate_batch


def sigmoid(x: ndarray):
    # return 1 / (1 + np.exp(-x))
    # return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return expit(x)


def sigmoid_d(x_e: ndarray):
    return (np.exp(-x_e)) / (1 + np.exp(-x_e)) ** 2


def swish(x: ndarray):
    return x * sigmoid(x)


def swish_d(x: ndarray):
    return swish(x) + sigmoid(x) * (1 - swish(x))


def relu(x: ndarray):
    x[np.where(x < 0)] = 0
    return x


def relu_d(x: ndarray):
    x[np.where(x >= 0)] = 1
    x[np.where(x < 0)] = 0

    return x


def h_f(x: ndarray):
    return relu(x)


def h_df(x_e: ndarray):
    return relu_d(x_e)


def o_f(x: ndarray):
    return sigmoid(x)


def o_df(x_e: ndarray):
    return sigmoid_d(x_e)


def mse(tar: ndarray, out: ndarray):
    return np.sum((tar - out) ** 2) / out.size


class MLP:
    def __init__(self, winner_cnt: int = 1):
        self.winner_cnt = winner_cnt

        self.l_inp = np.zeros(128)
        self.l_h = np.zeros(256)
        self.l_o = np.zeros(2)

        self.w_ih: ndarray = np.random.rand(self.l_inp.size, self.l_h.size) * 0.001
        self.w_ho: ndarray = np.random.rand(self.l_h.size, self.l_o.size) * 0.001

    def infer_one(self, inp: ndarray):
        self.l_inp = np.copy(inp)

        self.l_h = h_f(np.dot(self.l_inp, self.w_ih))

        self.lateral_inhibition()

        self.l_o = o_f(np.dot(self.l_h, self.w_ho))
        self.get_confidence()
        return self.l_o

    def lateral_inhibition(self):
        winner_idx_arr = np.argsort(self.l_h)[::-1]
        self.l_h[winner_idx_arr[self.winner_cnt:]] = 0

    def train(self, batch: List[Any], epoch_cnt: int = 100, lr0: float = 0.01,
              push_delta: float = 0.01,
              wta_lambda: float = 0.1, ):
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

                # uw_norm = np.linalg.norm(uw_ih)
                # if uw_norm != 0:
                #     dw_ih += uw_ih / uw_norm

                # dw_ih += uw_ih
                self.wta_train(dw_ih, dw_ho, push_delta, wta_lambda)

            self.w_ih -= lr * (dw_ih / len(batch))
            self.w_ho -= lr * (dw_ho / len(batch))

            self.get_confidence()

    def wta_train(self, dw_ih: ndarray, dw_ho: ndarray, push_delta: float, wta_lambda: float):
        l_h = np.dot(self.l_inp, self.w_ih)
        # l_h = self.l_h

        winner_idx_arr = np.argsort(l_h)[::-1]
        pull_idx_arr = winner_idx_arr[0:1]
        push_idx_arr = winner_idx_arr[1:]

        for pull_idx in pull_idx_arr:
            u_w_ih = wta_lambda * (self.l_inp - self.w_ih.T[pull_idx] * l_h[pull_idx])

            uw_norm = np.linalg.norm(u_w_ih)
            if uw_norm == 0:
                continue

            u_w_ih = u_w_ih / uw_norm
            dw_ih.T[pull_idx] += u_w_ih

        for push_idx in push_idx_arr:
            u_w_ih = wta_lambda * (self.l_inp - self.w_ih.T[push_idx] * l_h[push_idx]) * -push_delta

            uw_norm = np.linalg.norm(u_w_ih)
            if uw_norm == 0:
                continue

            u_w_ih = u_w_ih / uw_norm
            dw_ih.T[push_idx] += u_w_ih

        # winner_idx_arr = np.argsort(self.l_o)[::-1]
        # pull_idx = winner_idx_arr[0]
        # push_idx = winner_idx_arr[1]
        #
        # dw_ho.T[pull_idx] += wta_lambda * (self.l_h - self.w_ho.T[pull_idx] * self.l_o[pull_idx])
        # dw_ho.T[push_idx] += wta_lambda * (self.l_h - self.w_ho.T[push_idx] * self.l_o[push_idx]) * -push_delta

        return dw_ih, dw_ho

    def test(self, batch: List[Any]):
        error = 0

        for inp, target_out in batch:
            out = self.infer_one(inp)
            error += mse(target_out, out)

        error /= len(batch)
        print(f'mse: {error}')

        return error

    def get_confidence(self):
        confidence = np.sum(np.abs(self.l_h[np.newaxis].T - self.l_h))
        print(f'{confidence=}')

        return confidence


if __name__ == '__main__':
    cls_size = 5

    mlp = MLP(winner_cnt=16)
    train_batch, test_batch = generate_batch(ns_clstr=[cls_size, cls_size], cluster_std=0.04, n_features=mlp.l_inp.size)

    mse_start = mlp.test(test_batch)

    mlp.train(batch=train_batch, epoch_cnt=100, lr0=0.04, push_delta=0.4, wta_lambda=0.01)

    mse_trained = mlp.test(test_batch)

    print(f'{mse_start / mse_trained if mse_trained > 0 else 0}')
