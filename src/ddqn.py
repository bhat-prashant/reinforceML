#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDQN():

    def __init__(self, state_size, action_size, seed):
        """ DDQN algorithm implementation
        https://arxiv.org/abs/1509.06461

        :param state_size: int,
            size of state space representation, input for neural network
        :param action_size: int,
            size of action space representation, output for neural network
        :param seed: int,
            random seed
        """

        self.t_step = 0
        self._buffer_size = int(1e5)
        self._batch_size = 64
        self._gamma = 0.99
        self._TAU = 1e-3
        self._LR = 5e-4
        self._update_frequency = 8
        self._eps = 1.0
        self._eps_end = 0.01
        self._eps_decay = 0.995
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self._LR)
        self.memory = ReplayBuffer(action_size, self._buffer_size, self._batch_size, seed)

    def perform_action(self, state, use_rl=True):
        """ Action for performing mutation on the current individual

        :param state: ndarray,
            current state
        :param use_rl: boolean,
            Whether to use reinforcement learning in evolution or not.
        :return: int,
            index of an action in action space
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > self._eps and use_rl:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def timestep(self, state, action, reward, next_state):
        """ 1. For each mutation, add an entry into replay memory
        2. If there are enough samples in the memory, train q-network

        :param state: ndarray,
            current state
        :param action: ndarray,
            current action
        :param reward: int
        :param next_state: ndarray,
            next state
        :return: None
        """
        self.memory.add(state, action, reward, next_state)
        self.t_step = (self.t_step + 1) % self._update_frequency
        if self.t_step == 0:
            if len(self.memory) > self._batch_size:
                experiences = self.memory.sample()
                self.train_ddqn(experiences, self._gamma)

    def train_ddqn(self, experiences, gamma):
        """ Update neural networks - q-network and target network

        :param experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples
        :param gamma: discount factor
        :return:
        """
        states, actions, rewards, next_states = experiences

        # DDQN - compute q value from max_action of q-network
        max_actions = self.qnetwork_target(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, max_actions)

        # DQN - compute q value from max_action of target network
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next)
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon for epsilon greedy policy
        self._eps = max(self._eps_end, self._eps_decay * self._eps)
        self.update_target_network(self.qnetwork_local, self.qnetwork_target, self._TAU)

    def update_target_network(self, local_model, target_model, tau):
        """ Update target network with values form q-network

        :param local_model: q-network
        :param target_model: target network
        :param tau:
        :return: None
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Replay Buffer

        :param action_size: int,
            size of action space
        :param buffer_size: int,
            buffer size
        :param batch_size: int,
            batch size for random sampling of buffer
        :param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        """ Add an entry into replay buffer

        :param state: ndarray
        :param action: ndarray
        :param reward: int
        :param next_state: ndarray
        :return: None
        """
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """" Sample a batch of historical transitions from the replay memory
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        return (states, actions, rewards, next_states)

    def __len__(self):
        """ returns length of the replay memory

        :return: int
        """
        return len(self.memory)


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """ Neural networks for DDQN

        :param state_size: int,
            size of state space for neural network input
        :param action_size: int,
            size of action space for neural network output
        :param seed: int,
            random seed
        :param fc1_units: int,
            Number of nodes in first hidden layer
        :param fc2_units: int,
            Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """ Builds neural networks

        :param state:
        :return:
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
