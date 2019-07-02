from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
from six.moves import range
from six.moves import zip

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(object):

    def __init__(self,
                 input_shape,
                 q_fn,
                 learning_rate=0.001,
                 learning_rate_decay_step=10000,
                 learning_rate_decay_rate=0.8,
                 optimizer='Adam',
                 grad_clipping=None,
                 gamma=1.0,
                 epsilon=0.2,
                 double_q=True,
                 num_bootstrap_heads=10,
                 scope='dqn',
                 reuse=None):
        self.input_shape = input_shape
        self.q_fn = q_fn
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_step
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.optimizer = optimizer
        self.grad_clipping = grad_clipping
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_bootstrap_heads = num_bootstrap_heads
        self.double_q = double_q
        self.scope = scope
        self.reuse = reuse


def get_hparams(*args):
    hparams = {
        'atom_types': ['C', 'O', 'N'],
        'max_steps_per_episode': 40,
        'allow_removal': True,
        'allow_no_modification': True,
        'allow_bonds_between_rings': False,
        'allowed_ring_sizes': [3, 4, 5, 6],
        'replay_buffer_size': 1000000,
        'learning_rate': 1e-4,
        'learning_rate_decay_steps': 10000,
        'learning_rate_decay_rate': 0.8,
        'num_episodes': 5000,
        'batch_size': 64,
        'learning_frequency': 4,
        'update_frequency': 20,
        'grad_clipping': 10.0,
        'gamma': 0.9,
        'double_q': True,
        'num_bootstrap_heads': 12,
        'prioritized': False,
        'prioritized_alpha': 0.6,
        'prioritized_beta': 0.4,
        'prioritized_epsilon': 1e-6,
        'fingerprint_radius': 3,
        'fingerprint_length': 2048,
        'dense_layers': [1024, 512, 128, 32],
        'activation': 'relu',
        'optimizer': 'Adam',
        'batch_norm': False,
        'save_frequency': 1000,
        'max_num_checkpoints': 100,
        'discount_factor': 0.7
    }
    return hparams.update(args)


class MultiLayerNetwork(nn.Module):

    def __init__(self, hparams):
        self.hparams = hparams
        self.dense = nn.Sequential()

        hparams_layers = self.hparams['dense_layers']
        hparams_layers = [self.hparams['fingerprint_length']] + \
                         hparams_layers + \
                         [self.hparams['num_bootstrap_heads']]
        for i, num_cells in enumerate(hparams_layers):
            self.dense.add_module('dense_%i' % i, nn.Linear(num_cells, hparams_layers[i + 1]))
            self.dense.add_module('%s_%i' %())

    def forward(self, x):
        pass


