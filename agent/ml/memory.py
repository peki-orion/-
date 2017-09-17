# -*- coding: utf-8 -*-

"""
based on https://github.com/ISakony/NEC_chainerrl_CartPole-v0
"""

from __future__ import (
    division,
    print_function,
    unicode_literals,
    absolute_import
)
from builtins import *  # noqa
from future import standard_library

import numpy as np

standard_library.install_aliases()


class Ma_memory(object):
    def __init__(self, n_of_action, limit_n_of_memory, embedding_dim):
        self.limit_n_of_memory = limit_n_of_memory
        self.m = 1
        self.maq = self.first_memory_q(n_of_action)
        self.maq_c = self.m
        self.mah = self.first_memory_key(n_of_action)
        self.mah_c = self.m
        self.embedding_dim = embedding_dim

    def first_memory_q(self, n_of_action):
        mlist = []
        for ii in range(n_of_action):
            ml = np.zeros((self.limit_n_of_memory, 1), dtype=np.float32)
            for i in range(self.m):
                mm = np.random.rand(1)
                mm = mm.astype(np.float32)
                ml[0] = mm
            mlist.append(ml)
        return mlist

    def first_memory_key(self, n_of_action):
        mlist = []
        for ii in range(n_of_action):
            ml = np.zeros((self.limit_n_of_memory, 128), dtype=np.float32)  # TODO
            for i in range(self.m):
                mm = np.random.rand(128)  # TODO
                mm = mm.astype(np.float32)
                ml[0] = mm
            mlist.append(ml)
        return mlist

    def add_memory(self, h, q, greedy_action, dist, ind, alpha):
        if self.mah_c == self.limit_n_of_memory + 1:
                # print("change")
                self.mah[greedy_action][ind[greedy_action][0][0]] = h[0]
                self.maq[greedy_action][ind[greedy_action][0][0]] = q[0]

        elif dist[greedy_action][0][0] == 0.0:
                # print("update", self.maq[greedy_action][ind[greedy_action][0][0]],
                #       alpha * (q[0] - self.maq[greedy_action][ind[greedy_action][0][0]]))
                self.maq[greedy_action][ind[greedy_action][0][0]] = self.maq[greedy_action][ind[greedy_action][0][0]] + alpha * (q[0] - self.maq[greedy_action][ind[greedy_action][0][0]])
        else:

            self.mah[greedy_action][self.mah_c - 1] = h[0]
            self.mah_c += 1

            self.maq[greedy_action][self.maq_c - 1] = q[0]
            self.maq_c += 1


class D_memory(object):
    def __init__(self, limit_n_of_memory):
        self.limit_n_of_memory = limit_n_of_memory
        self.mem = []

    def add_memory(self, state, action, q, ind):
        if len(self.mem) == self.limit_n_of_memory:
            self.mem = self.replace(state, action, q, ind)
        else:
            self.mem.append((state, action, q, ind))

    def replace(self, state, action, q, ind):
        mem_re = self.mem[1:-1]
        mem_re.append((state, action, q, ind))
        return mem_re


class N_step_memory(object):
    def __init__(self, gamma, N_horizon, ma_memory, d_memory, alpha):
        self.mem = []
        self.gamma = gamma
        self.N_horizon = N_horizon
        self.ma_memory = ma_memory
        self.d_memory = d_memory
        self.alpha = alpha

    def add_replace_memory(self, state, action, key, reward, t_step, done, q_ontime, dist, ind):
        self.mem.append([state, action, key, reward, t_step, dist, ind])

        '''
        mem.append([state,action,key,qn,t_step,dist,ind])
        mem[i][0] state
        mem[i][1] action
        mem[i][2] h = key
        mem[i][3] qn = sum reward from now-step to i-step
        mem[i][4] count steps
        '''

        if done:
            for i in range(len(self.mem)):
                self.mem[i][4] = self.mem[i][4] + 1  # n_step +1
                self.mem[i][3] += (self.gamma ** (self.mem[i][4] - 1)) * reward

                t_qn = np.ndarray((1, 1), dtype=np.float32)
                t_qn[0] = self.mem[i][3]  # qn

                self.d_memory.add_memory(self.mem[i][0], self.mem[i][1], t_qn, self.mem[i][6])
                self.ma_memory.add_memory(
                    self.mem[i][2], t_qn, self.mem[i][1], self.mem[i][5], self.mem[i][6], self.alpha)
