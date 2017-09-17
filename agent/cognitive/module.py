# coding: utf-8

import os

import brica1.gym
import numpy as np
import six.moves.cPickle as pickle

from ml.cnn_feature_extractor import CnnFeatureExtractor
from ml.agent import Agent

from config.model import CNN_FEATURE_EXTRACTOR, CAFFE_MODEL, MODEL_TYPE

from config.log import APP_KEY
import logging
app_logger = logging.getLogger(APP_KEY)

use_gpu = int(os.getenv('GPU', '-1'))


class VVCComponent(brica1.Component):
    image_feature_count = 1
    cnn_feature_extractor = CNN_FEATURE_EXTRACTOR
    model = CAFFE_MODEL
    model_type = MODEL_TYPE
    image_feature_dim = 256 * 6 * 6

    def __init__(self, n_output=10240, n_input=1):
        # image_feature_count = 1
        super(VVCComponent, self).__init__()

        self.use_gpu = use_gpu
        self.n_output = n_output
        self.n_input = n_input

    def set_model(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def load_model(self, cnn_feature_extractor):
        if os.path.exists(cnn_feature_extractor):
            app_logger.info("loading... {}".format(cnn_feature_extractor))
            self.feature_extractor = pickle.load(open(cnn_feature_extractor))
            app_logger.info("done")
        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type,
                                                         self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(cnn_feature_extractor, 'w'))
            app_logger.info("pickle.dump finished")

    def fire(self):
        observation = self.get_in_port('Isocortex#V1-Isocortex#VVC-Input').buffer
        obs_array = self.feature_extractor.feature(observation, self.image_feature_count)

        self.results['Isocortex#VVC-BG-Output'] = obs_array
        self.results['Isocortex#VVC-UB-Output'] = obs_array


class BGComponent(brica1.Component):
    def __init__(self, n_input=10240, n_output=1):
        super(BGComponent, self).__init__()
        self.use_gpu = use_gpu
        self.epsilon = 1.0
        epsilon_decay_steps = 10 ** 4
        min_eps = 0.1
        actions = [0, 1, 2]
        self.input_dim = n_input
        self.embedding_dim = 128
        self.replay_size = 32

        self.agent = Agent(self.use_gpu, self.epsilon, min_eps, epsilon_decay_steps,
                           self.input_dim, self.embedding_dim, len(actions), self.replay_size)

        self.episode = 0

    def start(self):
        self.get_in_port('Isocortex#VVC-BG-Input').buffer
        action = self.agent.sample_action_space()
        return action

    def end(self, reward):  # Episode Terminated
        app_logger.info('episode finished. Reward:{:.1f} / Epsilon:{:.6f}'.format(reward, self.epsilon))
        reward = self.get_in_port('RB-BG-Input').buffer
        features = self.get_in_port('Isocortex#VVC-BG-Input').buffer
        self.get_in_port('UB-BG-Input').buffer

        action, ind, dist, key, eps = self.agent.step(features, self.episode)
        self.agent.update_memory_and_train(features, action, ind, dist, reward, key, True, self.episode)
        self.episode = 1

    def fire(self):
        reward = self.get_in_port('RB-BG-Input').buffer
        features = self.get_in_port('Isocortex#VVC-BG-Input').buffer
        self.get_in_port('UB-BG-Input').buffer

        action, ind, dist, key, eps = self.agent.step(features, self.episode)
        self.agent.update_memory_and_train(features, action, ind, dist, reward, key, False, self.episode)

        app_logger.info('Step:{}  Action:{}  Reward:{:.1f}  Epsilon:{:.6f}  Q_max:{:3f}'.format(
            self.agent.t, action, reward[0], eps, 1.0  # q_max  # TODO
        ))

        self.epsilon = eps
        self.results['BG-Isocortex#FL-Output'] = np.array([action])


class UBComponent(brica1.Component):
    def __init__(self):
        super(UBComponent, self).__init__()
        hist_size = 1
        dim = 10240
        vvc_input = np.zeros((hist_size, dim), dtype=np.uint8)
        self.last_state = vvc_input
        self.state = vvc_input
        self.time = 0

    def end(self, action, reward):
        self.time += 1
        self.results['UB-BG-Output'] = [None, None, None, None, None, None]

    def fire(self):
        self.state = self.get_in_port('Isocortex#VVC-UB-Input').buffer
        action, reward = self.get_in_port('Isocortex#FL-UB-Input').buffer

        self.results['UB-BG-Output'] = [None, None, None, None, None, None]
        self.last_state = self.state.copy()
        self.time += 1


class FLComponent(brica1.Component):
    def __init__(self):
        super(FLComponent, self).__init__()
        self.last_action = np.array([0])

    def fire(self):
        action = self.get_in_port('BG-Isocortex#FL-Input').buffer
        reward = self.get_in_port('RB-Isocortex#FL-Input').buffer
        self.results['Isocortex#FL-MO-Output'] = action
        self.results['Isocortex#FL-UB-Output'] = [self.last_action, reward]

        self.last_action = action
