import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from abc import *


class CircularBuffer:
    def __init__(self, maxlen):
        self.buffer = [None] * maxlen  # Pre-allocate fixed size
        self.maxlen = maxlen
        self.start = 0  # Points to the oldest element
        self.size = 0   # Number of elements in the buffer

    def append(self, value):
        # Compute the index where the new element will be added
        idx = (self.start + self.size) % self.maxlen
        self.buffer[idx] = value

        if self.size < self.maxlen:
            # Increment size if not full
            self.size += 1
        else:
            # Move start forward if full
            self.start = (self.start + 1) % self.maxlen

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        return self.buffer[index]


class BaseBuffer(metaclass=ABCMeta):
    def __init__(self, seed=0):
        np.random.seed(seed)
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def store(self, obs: np.ndarray, action: int, reward: float, done: bool, next_obs: np.ndarray):
        pass
    
    def sample(self, batch_size: int) -> dict:
        pass

