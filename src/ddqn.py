#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

from collections import deque

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop


class DDQN:
    def __init__(self, input_units):
        self.q_network = self.get_model(input_units)
        self.target_network = self.get_model(input_units)

    def get_model(self, input_units):
        model = Sequential()
        model.add(Dense(units=128, input_shape=input_units, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.10))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.10))
        model.add(Dense(units=input_units, kernel_initializer='uniform', activation='softmax'))
        model.compile(optimizer=RMSprop(lr=0.01), )
        return model

    def experience_replay(self):
        pass

    def loss(self, y_i, q_i):
        # huber loss
        pass


class Replay_Memory():
    def __init__(self, random_state, max_size=1000):
        self.memory = deque()
        self.max_size = max_size
        self.random_state = random_state

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.max_size:
            self.memory.popleft()

    def mini_batch(self, n=32):
        mini_batch = self.random_state.choice(self.memory, size=n)
        return mini_batch

    def size(self):
        return len(self.memory)

    def is_full(self):
        return True if self.size() >= self.max_size else False
