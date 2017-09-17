# -*- coding: utf-8 -*-

import chainer
import numpy as np

from chainer.optimizers import RMSpropGraves, Adam

from nec import (
    QFunction,
    LinearDecayEpsilonGreedy,
    NEC,
)


class Agent(object):
    """Machine learning module interface"""

    def __init__(self, use_gpu, eps_st, eps_end, eps_decay_step, input_dim, embedding_dim, n_actions, batch_size):
        self.use_gpu = use_gpu
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_hidden_channels = 64
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = 0.99
        self.phi = lambda x: x.astype(np.float32, copy=False)
        self.t = 0

        self.q_func = QFunction(self.use_gpu, self.input_dim, self.embedding_dim, self.n_actions, self.num_hidden_channels)
        self.q_func.to_cpu()

        self.explorer = LinearDecayEpsilonGreedy(eps_st, eps_end, eps_decay_step, self.sample_action_space)
        self.optimizer = RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)

        # self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.q_func)

        self.agent = NEC(self.use_gpu, self.q_func, self.optimizer, batch_size,
                         self.gamma, self.explorer, self.n_actions, False, self.embedding_dim)

    def step(self, obs, episode):
        action, ind, dist, key = self.agent.act(obs, episode)
        self.t += 1
        return action, ind, dist, key, self.explorer.epsilon

    def update_memory_and_train(self, obs, action, ind, dist, reward, key, done, episode):
        self.agent.append_memory_and_train(obs, action, ind, dist, reward, key, done, episode)

    def sample_action_space(self):
        # Uniformly randomly sample a random element of this action space
        # see also https://github.com/openai/gym/blob/master/gym/core.py#L222
        return np.random.randint(0, self.n_actions)
