import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.nn import Linear
from .base import BasePolicy

class MlpPolicy(BasePolicy):
    name = 'ddqn'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 action_size,
                 duel,
                 width,
                 activation,
                 normalization):
        super().__init__()
        self.action_size = action_size
        hid_dim = hid_dim * width

        self.fc = nn.Sequential(
            Linear(in_dim, hid_dim),
            nn.ReLU(), 
            Linear(hid_dim, action_size)
        )

    def forward(self, x, log=False):
        
        q = self.fc(x)
        # (batch_size, action_size)
        q = q.view(-1, self.action_size)
        info = {}
        
        return q, info