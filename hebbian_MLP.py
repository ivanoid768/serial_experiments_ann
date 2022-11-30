from typing import List, Any

import numpy as np
from numpy import ndarray

from MLP import relu, relu_d, sigmoid, sigmoid_d
from datapoint_generation import generate_batch


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

        self.avg_confidence = []
        self.prev_confidence = 0

    def infer_forward(self, inp: ndarray):
        self.l_inp = np.copy(inp)

        self.l_h = h_f(np.dot(self.l_inp, self.w_ih))
        self.l_o = o_f(np.dot(self.l_h, self.w_ho))

        return self.l_o

    def infer_backward(self, output: ndarray):
        self.l_o += output
        # self.l_o = output

        self.l_h += h_f(np.dot(self.l_o, self.w_ho.T))

        l_inp = (np.dot(self.l_h, self.w_ih.T))
        self.l_inp += l_inp

        return l_inp

    def train_h(self, push_delta: float = 0.04, winner_cnt: int = 16):
        l_h = self.l_h

        winner_idx_arr = np.argsort(l_h)[::-1]
        pull_idx_arr = winner_idx_arr[0:1]
        push_idx_arr = winner_idx_arr[1:1 + int(winner_cnt / 2)]

        dw_ih = self.get_delta_w(l_h, pull_idx_arr, push_delta, push_idx_arr, self.w_ih, self.l_inp)
        dw_ho = self.get_delta_w(l_h, pull_idx_arr, push_delta, push_idx_arr, self.w_ho.T, self.l_o)

        return dw_ih, dw_ho

    def train_o(self, push_delta: float = 0.04, winner_cnt: int = 1):
        layer = self.l_o

        winner_idx_arr = np.argsort(layer)[::-1]
        pull_idx_arr = winner_idx_arr[0:1]
        push_idx_arr = winner_idx_arr[1:1 + int(winner_cnt / 2)]

        dw_ho = self.get_delta_w(layer, pull_idx_arr, push_delta, push_idx_arr, self.w_ho, self.l_h)

        return dw_ho

    def train_inp(self, push_delta: float = 0.04, winner_cnt: int = 8):
        layer = self.l_inp

        winner_idx_arr = np.argsort(layer)[::-1]
        pull_idx_arr = winner_idx_arr[0:1]
        push_idx_arr = winner_idx_arr[1:1 + int(winner_cnt / 2)]

        dw_ih = self.get_delta_w(layer, pull_idx_arr, push_delta, push_idx_arr, self.w_ih.T, self.l_h)

        return dw_ih

    @staticmethod
    def get_delta_w(layer, pull_idx_arr, push_delta, push_idx_arr, w: ndarray, l_inp: ndarray):
        dw = np.zeros(w.shape)

        for pull_idx in pull_idx_arr:
            u_w_ih = l_inp - w.T[pull_idx] * layer[pull_idx]

            uw_norm = np.linalg.norm(u_w_ih)
            if uw_norm == 0:
                continue

            u_w_ih = u_w_ih / uw_norm
            dw.T[pull_idx] += u_w_ih

        for push_idx in push_idx_arr:
            u_w_ih = (l_inp - w.T[push_idx] * layer[push_idx]) * -push_delta

            uw_norm = np.linalg.norm(u_w_ih)
            if uw_norm == 0:
                continue

            u_w_ih = u_w_ih / uw_norm
            dw.T[push_idx] += u_w_ih

        return dw

    @staticmethod
    def loss(l_o: ndarray, out: ndarray, inp: ndarray, l_inp: ndarray):
        return (np.sum((l_o - out) ** 2) / l_o.size + np.sum((l_inp - inp) ** 2) / l_inp.size) / 2

    def train_one(self, inp: ndarray, out: ndarray, epoch_cnt: int = 100, lr0: float = 0.01, push_delta: float = 0.04):
        l_o = self.infer_forward(inp)
        l_inp = self.infer_backward(out)
        start_err = self.loss(l_o, out, inp, l_inp)
        print(f'{start_err=}')

        dw_ho = self.train_o(push_delta)
        dw_ih, dw_ho_2 = self.train_h(push_delta)
        dw_ih_2 = self.train_inp(push_delta)

        # update weights
        self.w_ih += -lr0 * (dw_ih + dw_ih_2.T) / 2
        self.w_ho += -lr0 * (dw_ho + dw_ho_2.T) / 2

        l_o = self.infer_forward(inp)
        l_inp = self.infer_backward(out)
        err = self.loss(l_o, out, inp, l_inp)
        print(f'{err=} {err - start_err=}')

    def lateral_inhibition(self):
        winner_idx_arr = np.argsort(self.l_h)[::-1]
        self.l_h[winner_idx_arr[self.winner_cnt:]] = 0

    def train(self, batch: List[Any], epoch_cnt: int = 100, lr0: float = 0.01,
              push_delta: float = 0.01,
              wta_lambda: float = 0.1, ):
        self.avg_confidence = []

        avg_error = 0
        for epoch_idx in range(epoch_cnt):
            lr = (epoch_cnt - epoch_idx) * lr0

            # dw_ih: ndarray = np.zeros(self.w_ih.shape)
            # dw_ho: ndarray = np.zeros(self.w_ho.shape)
            # for inp, target_out in batch:
            #     self.infer_one(inp)
            #     # self.wta_train(dw_ih, dw_ho, push_delta, wta_lambda)
            #
            # self.update_weights(batch, dw_ho, dw_ih, lr)

            dw_ih: ndarray = np.zeros(self.w_ih.shape)
            dw_ho: ndarray = np.zeros(self.w_ho.shape)
            for inp, target_out in batch:
                out = self.infer_one(inp)

                o_e = (out - target_out) * o_df(self.l_o)
                dw_ho += np.dot(self.l_h[np.newaxis].T, o_e[np.newaxis])

                h_e = np.dot(o_e, self.w_ho.T) * h_df(self.l_h)
                uw_ih = np.dot(self.l_inp[np.newaxis].T, h_e[np.newaxis])

                uw_norm = np.linalg.norm(uw_ih, axis=0)
                if np.where(uw_norm != 0)[0].size >= uw_ih.shape[1]:
                    dw_ih += uw_ih / uw_norm

                dw_ih += uw_ih

                avg_error += mse(target_out, out)

            self.update_weights(batch, dw_ho, dw_ih, lr)

            self.get_avg_confidence()

            avg_error /= len(batch)
            print(f'{epoch_idx=} {avg_error=}')

            if avg_error == 0.0:
                break

            avg_error = 0

    def update_weights(self, batch, dw_ho, dw_ih, lr):
        self.w_ih -= lr * (dw_ih / len(batch))
        self.w_ho -= lr * (dw_ho / len(batch))

    def wta_train(self, dw_ih: ndarray, dw_ho: ndarray, push_delta: float, wta_lambda: float):
        l_h = np.dot(self.l_inp, self.w_ih)
        # l_h = self.l_h

        winner_idx_arr = np.argsort(l_h)[::-1]
        pull_idx_arr = winner_idx_arr[0:1]
        push_idx_arr = winner_idx_arr[1:1 + int(self.winner_cnt / 2)]

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
        # print(f'{confidence=}
        self.avg_confidence.append(confidence)

        return confidence

    def get_avg_confidence(self):
        avg_confidence = np.mean(np.array(self.avg_confidence))
        self.avg_confidence = []

        print(f'{avg_confidence=} {(avg_confidence - self.prev_confidence > 0)=}')
        self.prev_confidence = avg_confidence
        return avg_confidence


if __name__ == '__main__':
    cls_size = 5

    mlp = MLP(winner_cnt=16)
    train_batch, test_batch = generate_batch(ns_clstr=[cls_size, cls_size], cluster_std=0.04, n_features=mlp.l_inp.size)

    mlp.train_one(train_batch[0][0], train_batch[0][1], epoch_cnt=100, lr0=0.01, push_delta=0.04)

    # mse_start = mlp.test(test_batch)
    #
    # mlp.train(batch=train_batch, epoch_cnt=100, lr0=0.04, push_delta=0.4, wta_lambda=0.01)
    #
    # mse_trained = mlp.test(test_batch)
    #
    # print(f'{mse_start / mse_trained if mse_trained > 0 else 0}')
