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

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pickle

from chainer import cuda
from chainer.functions.loss import mean_squared_error
from sklearn.neighbors.kd_tree import KDTree

from ml.memory import (
    D_memory,
    Ma_memory,
    N_step_memory
)

standard_library.install_aliases()


def get_greedy_actions(q_values):
    # substitute function of chainerrl.DiscreteActionValue.greedy_actions()
    # see also http://chainerrl.readthedocs.io/en/latest/_modules/chainerrl/action_value.html#DiscreteActionValue
    return chainer.Variable(q_values.data.argmax(axis=1).astype(np.int32))


def select_action_epsilon_greedily(epsilon, random_action_func,
                                   greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class LinearDecayEpsilonGreedy(object):
    """Epsilon-greedy with linearyly decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)
        # greedy_str = 'greedy' if greedy else 'non-greedy'
        return a


class QFunction(chainer.Chain):
    def __init__(self, use_gpu, input_dim, embedding_dim, n_actions, n_hidden_channels=64):
        self.use_gpu = use_gpu
        self.q_list = np.ndarray((1, n_actions), dtype=np.float32)  # for discreteActionValue
        super(QFunction, self).__init__(
            l0=L.Linear(input_dim, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, embedding_dim))

    def __call__(self, x, ma):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))

        # kd_tree
        q_train = []  # for train [variable,variable]
        ind_list = []  # for train
        dist_list = []  # for train
        for j in range(len(ma.maq)):  # loop n_actions
            h_list = ma.mah[j]
            lp = len(h_list)
            leaf_size = lp + (lp / 2)

            tree = KDTree(h_list, leaf_size=leaf_size)
            h_ = h.data

            if lp < 50:
                k = lp
            else:
                k = 50
            dist, ind = tree.query(h_, k=k)

            mahi = ma.mah[j][ind[0]]
            hi = chainer.Variable(cuda.to_cpu(mahi))
            tiled_h = chainer.Variable(np.tile(h.data, (len(ind[0]), 1)))
            wi = F.expand_dims(1 / (F.sqrt(F.sum((tiled_h - hi) * (tiled_h - hi), axis=1) + 1e-3)), 1)
            w = F.sum(wi, axis=0)
            maqi = ma.maq[j][ind[0]]
            q = chainer.Variable(cuda.to_cpu(maqi))
            qq = F.expand_dims(F.sum(wi * q, axis=0) / w, 1)

            q_train.append(qq)
            ind_list.append(ind)
            dist_list.append(dist)

            self.q_list[0][j] = qq.data
        if self.use_gpu:
            qa = chainer.Variable(cuda.to_cpu(self.q_list))
        else:
            qa = self.q_list

        return qa, q_train, ind_list, dist_list, h.data
        # return chainerrl.action_value.DiscreteActionValue(qa), q_train, ind_list, dist_list, h.data


class NEC(object):
    def __init__(self, use_gpu, q_function, optimizer, batch_size, gamma, explorer, n_actions, retrain, embedding_dim):
        self.model = q_function
        self.n_actions = n_actions
        self.N_horizon = 100
        self.phi = lambda x: x
        self.update_frequency = 1
        self.minibatch_size = batch_size
        self.size_of_memory = 100000  # 5*10^5
        self.embedding_dim = embedding_dim

        if not retrain:
            self.ma_memory = Ma_memory(n_actions, self.size_of_memory, self.embedding_dim)
        else:
            self.ma_memory = Ma_memory(n_actions, self.size_of_memory, self.embedding_dim)
            self.ma_memory.maq = pickle.load(open("Ma_memory_maq_.pickle", "rb"))
            self.ma_memory.mah = pickle.load(open("Ma_memory_mah_.pickle", "rb"))

        self.d_memory = D_memory(self.size_of_memory)
        self.alpha = 0.1  # learning late
        self.n_step_memory = N_step_memory(gamma, self.N_horizon, self.ma_memory,
                                           self.d_memory, self.alpha)
        self.optimizer = optimizer
        self.gamma = gamma  # 0.99
        self.explorer = explorer
        self.n_step = 0
        self.q_ontime = None  #
        self.q_target = None
        self.t = 0
        self.average_loss_all = np.array(0.0).astype('f')
        self.average_loss_c = np.array(0.0).astype('f')
        self.average_loss = np.array(0.0).astype('f')
        self.use_gpu = use_gpu

    def act(self, state, episode):  #
        self.n_step += 1
        if self.n_step == self.N_horizon + 1:
            self.n_step = 1

        # action
        with chainer.no_backprop_mode():
            action_value = self.model(F.expand_dims(state, 0), self.ma_memory)

        greedy_action = cuda.to_cpu(get_greedy_actions(action_value[0]).data)[0]

        if episode < 1:
            action = np.random.randint(0, self.n_actions)
        else:
            action = self.explorer.select_action(
                self.t, lambda: greedy_action, action_value=action_value[0])
        self.t += 1

        # print(action_value[1])
        self.q_ontime = cuda.to_cpu(action_value[1][greedy_action].data)

        self.last_action = action
        ind = action_value[2]
        dist = action_value[3]
        key = cuda.to_cpu(action_value[4])

        return self.last_action, ind, dist, key

    def append_memory_and_train(self, state, action, ind, dist, reward, key, done, episode):
        t_step = 1
        self.n_step_memory.add_replace_memory(state, action, key, reward,
                                              t_step, done, self.q_ontime, dist, ind)

        if len(self.d_memory.mem) < self.minibatch_size:
            n_loop = len(self.d_memory.mem)
        else:
            n_loop = self.minibatch_size

        if episode < 1:
            return

        if self.t % 4 == 0:  # 16
            loss = np.array(0.0).astype('f')
            for ii in range(n_loop):
                rnd = np.random.randint(len(self.d_memory.mem))
                obs, action, t_q, t_ind = self.n_step_memory.d_memory.mem[rnd]
                obs = chainer.Variable(cuda.to_cpu(obs))
                t_q = chainer.Variable(cuda.to_cpu(t_q))
                tav = self.model(F.expand_dims(obs, 0), self.n_step_memory.ma_memory)
                greedy_action = cuda.to_cpu(get_greedy_actions(tav[0]).data)[0]
                train_q = tav[1][greedy_action]

                loss += mean_squared_error.mean_squared_error(train_q, t_q)

                self.average_loss_all += loss
                self.average_loss_c += 1.0
                self.average_loss = self.average_loss_all / self.average_loss_c

            if n_loop != 0:
                loss /= n_loop
                self.model.zerograds()
                loss.backward()
                self.optimizer.update()

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """
        assert self.last_action is not None

        # Add a transition to d_memory
        self.stop_episode()

    def stop_episode(self):
        self.last_action = None
        self.q_ontime = None
        self.n_step = 0
        self.average_loss = 0.0
        self.average_loss_all = 0.0
        self.average_loss_c = 0.0

    def get_statistics(self):
        return [
            ('Ma', "q", len(self.n_step_memory.ma_memory.maq[0]), len(self.n_step_memory.ma_memory.maq[1])),
            ('D', len(self.n_step_memory.d_memory.mem)),
        ]
