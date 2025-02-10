import torch
import torch.nn as nn
import numpy as np
from collections import deque
from .base import BaseBuffer, CircularBuffer
from einops import rearrange

class NumpyExperience():
    def __init__(self, size, n_step, frame_stack=4):
        self.index = 0
        self.size = size
        self.n_step = n_step
        self.frame_stack = frame_stack
        self.obs_transitions = np.zeros(shape=(self.size + self.n_step + self.frame_stack, 1, 84, 84), dtype=np.uint8) # for saving the final n_step states
        self.transitions = CircularBuffer(maxlen=self.size)

    def append(self, data):
        """
        data = (obs, action, action_list, G, done, next_obs)
        """
        obs, action, action_list, G, done, next_obs = data
        obs_start_idx = self.index
        next_obs_start_idx = self.index + 1
        self.transitions.append((obs_start_idx, action, action_list, G, done, next_obs_start_idx))
        self.obs_transitions[self.index : self.index + self.frame_stack][:] = obs[:]
        self.obs_transitions[self.index + self.frame_stack][:] = next_obs[-1][:]
        self.index = (self.index + 1) % self.size  # Update index

    def get(self, data_idxs):
        """
        data = (obs, action, action_list, G, done, next_obs)
        """
        transitions = []
        for idx in data_idxs:
            obs_start_idx, action, action_list, G, done, next_obs_start_idx = self.transitions[idx]
            obs = self.obs_transitions[obs_start_idx:obs_start_idx+self.frame_stack]
            next_obs = self.obs_transitions[next_obs_start_idx:next_obs_start_idx+self.frame_stack]
            transitions.append((obs, action, action_list, G, done, next_obs))
        return transitions 


class NumpyBuffer(BaseBuffer):
    name = 'numpy_buffer'
    def __init__(self, size, n_step, gamma, device, seed=0):
        super().__init__(seed=seed)
        # Initialize
        self.size = size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.num_in_buffer = 0
        self.n_step_transitions = deque(maxlen=self.n_step)
        self.transitions = NumpyExperience(size, n_step, frame_stack=4)
    
    def _get_n_step_info(self):
        transitions = list(self.n_step_transitions)
        obs, action, _, _, _ = transitions[0]
        _, _, G, done, next_obs = transitions[-1]
        for _, _, _reward, _done, _next_obs in reversed(transitions[:-1]):
            G = _reward + self.gamma * G * (1-_done)    
            if _done:
                done, next_obs = _done, _next_obs

        action_list = []
        for _, act, _, _, _ in transitions:
            action_list.append(act)
        
        return (obs, action, action_list, G, done, next_obs)
    
    def store(self, obs, action, reward, done, next_obs):
        self.n_step_transitions.append((obs, action, reward, done, next_obs))
        if len(self.n_step_transitions) < self.n_step:
            return
        transition = self._get_n_step_info()
        # store new transition with maximum priority
        self.transitions.append(data=transition)
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)

    def _get_transitions(self, batch_size):
        idxs = np.random.randint(0, self.num_in_buffer, size=(batch_size,))
        transitions = self.transitions.get(idxs)
        return idxs, transitions
    
    def sample(self, batch_size, mode='train'):
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not have enough transitions to sample')
        idxs, transitions = self._get_transitions(batch_size)

        # encode transitions
        obs_batch, act_batch, act_list_batch, return_batch, done_batch, next_obs_batch = zip(*transitions)
        obs_batch = self.encode_obs(obs_batch)  
        act_batch = torch.LongTensor(act_batch).to(self.device)
        act_list_batch = torch.LongTensor(act_list_batch).to(self.device)
        return_batch = torch.FloatTensor(return_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        next_obs_batch = self.encode_obs(next_obs_batch)

        weights = [1.0] * batch_size
        weights = torch.FloatTensor(weights).to(self.device)

        batch = {
            'idxs': idxs,
            'obs': obs_batch,
            'act': act_batch,
            'act_list_batch': act_list_batch,
            'return': return_batch,
            'done': done_batch,
            'next_obs': next_obs_batch,
            'weights': weights,
            'prior_weight': None
        }

        return batch
    
    def encode_obs(self, obs, prediction=False):
        obs = np.array(obs).astype(np.float32)
        obs = obs / 255.0

        # prediction: batch-size: 1
        if prediction:
            obs = np.expand_dims(obs, 0)

        # in current form, time-step is fixed to 1
        obs = np.expand_dims(obs, 1)

        # n: batch_size
        # t: 1
        # f: frame_stack
        # c: channel (atari: 1, dmc: 3)
        # h: height
        # w: width
        n, t, f, c, h, w = obs.shape
        obs = torch.FloatTensor(obs).to(self.device)

        return obs