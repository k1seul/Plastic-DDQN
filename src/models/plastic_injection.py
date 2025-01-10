import copy
import torch
import torch.nn as nn
from .base import Model


class PlasticityInjectionModel(nn.Module):
    def __init__(self, backbone, head, policy):
        super().__init__() 
        self.network_class = Model
        self.backbone = copy.deepcopy(backbone)
        self.head = copy.deepcopy(head)
        self.policy = copy.deepcopy(policy)
        self.pos_networks = nn.ModuleList([self.network_class(backbone, head, policy)])
        self.neg_networks = nn.ModuleList([])

    def forward(self, x):
        pos_outs = torch.stack([network.forward(x)[0] for network in self.pos_networks], dim=0)
        pos_sum = pos_outs.sum(dim=0)
        if self.neg_networks:
            neg_outs = torch.stack([network.forward(x)[0] for network in self.neg_networks], dim=0)
            neg_sum = neg_outs.sum(dim=0) 
            q = pos_sum - neg_sum
        else:
            q = pos_sum
        info = {} 
        return q, info

    def plasticity_inject(self):
        pos_network = self.network_class(copy.deepcopy(self.backbone), copy.deepcopy(self.head), copy.deepcopy(self.policy))
        neg_network = self.network_class(copy.deepcopy(self.backbone), copy.deepcopy(self.head), copy.deepcopy(self.policy))
        self.pos_networks.append(pos_network)
        self.neg_networks.append(neg_network)
        # The only learnable network is the last positive network. 
        for network in (self.pos_networks[-2], self.neg_networks[-1]):
            for param in network.parameters():
                param.requires_grad = False
        
    def copy_online(self, online_model):
        self.pos_networks = copy.deepcopy(online_model.pos_networks)
        self.neg_networks = copy.deepcopy(online_model.neg_networks)
        for network in self.pos_networks:
            for param in network.parameters():
                param.requires_grad = False