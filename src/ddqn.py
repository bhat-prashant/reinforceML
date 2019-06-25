#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import random

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

random.seed(10)
from keras.losses import mse
import numpy as np

class DDQN:
    def __init__(self, state_size, random_state, action_size=None, batch_size=32, buffer_size=1000,
                 discount_factor=0.99, epsilon=0.3):
        self.state_size = state_size
        if action_size is None:
            self.action_size = state_size
        else:
            self.action_size = action_size
        self.q_network = self.get_model()
        self.target_network = self.get_model()
        self.batch_size = batch_size
        self.random_state = random_state
        self.buffer_size = buffer_size
        self.replay_memory = ReplayBuffer(random_state=self.random_state, max_size=self.buffer_size)
        self.time_step = 0
        self.update_frequency = 8
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_model(self):
        model = Sequential()
        model.add(Dense(units=128, input_shape=(self.state_size,), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.10))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.10))
        model.add(Dense(units=self.action_size, kernel_initializer='uniform', activation='softmax'))
        model.compile(optimizer=RMSprop(lr=0.01), loss=mse)
        return model

    def step(self, transition):
        self.replay_memory.add(transition=transition)
        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            if self.replay_memory.size > self.batch_size:
                mini_batch = self.replay_memory.mini_batch(self.batch_size)
                self.experience_replay(mini_batch)

    # Train NN using mini batch of experiences
    def experience_replay(self, mini_batch):
        mini_batch = np.array(mini_batch)
        states, actions, rewards, next_states = mini_batch
        q_targets_next = self.target_network.predict(next_states)
        q_targets = rewards + (self.discount_factor * q_targets_next)
        self.q_network.train_on_batch(states, q_targets)

    # Returns actions for given state as per current policy
    def sample_action(self, state, ):
        action_values = self.q_network.predict(state)
        return action_values


class ReplayBuffer():
    def __init__(self, random_state, max_size=1000):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.random_state = random_state

    def add(self, transition):
        self.buffer[self.index] = transition
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def mini_batch(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]
