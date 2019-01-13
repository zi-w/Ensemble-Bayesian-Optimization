#!/usr/bin/env python
import numpy as np
import scipy.cluster.hierarchy as hi


def sample_multinomial(prob, shape, dim_limit):
    assert isinstance(shape, int)
    prob = prob / np.sum(prob)
    ret = - np.ones(shape, dtype=np.int)
    for i in range(shape):
        cnt = 0
        while cnt < 100:
            assign = np.random.choice(len(prob), p=prob)
            if np.sum(ret == assign) < dim_limit:
                ret[i] = assign
                break
            cnt += 1
        if cnt >= 100:
            raise ValueError('Not able to sample multinomial with dim limit within 100 rounds.')
    return ret


def sample_categorical(prob):
    prob = prob / np.sum(prob)
    return np.random.choice(len(prob), p=prob)


def find(pred):
    return np.where(pred)[0]


def gumbel():
    return -np.log(-np.log(np.random.random()))


def mean_z(z_all, dim_limit):
    # use correlation clustering to average group assignments
    lz = hi.linkage(z_all.T, 'single', 'hamming')
    # not sure why cluster id starts from 1
    z = hi.fcluster(lz, 0) - 1
    all_cat = np.unique(z)
    for a in all_cat:
        a_size = np.sum(a == z)
        if a_size > dim_limit:
            z[a == z] = sample_multinomial([1.] * a_size, a_size, dim_limit)
    return z


class NormalizedInputFn:
    def __init__(self, fn_instance, x_range):
        self.fn_instance = fn_instance
        self.x_range = x_range

    def __call__(self, x):
        return self.fn_instance(self.project_input(x))

    def project_input(self, x):
        return x * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

    def inv_project_input(self, x):
        return (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

    def get_range(self):
        return np.array([np.zeros(self.x_range[0].shape[0]), np.ones(self.x_range[0].shape[0])])


class ConstantOffsetFn:
    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        return self.fn_instance(x) + self.offset

    def get_range(self):
        return self.fn_instance.get_range()
