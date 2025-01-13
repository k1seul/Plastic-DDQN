import copy
import torch
import torch.nn as nn
from .base import Model


class PlasticityInjectionModel(nn.Module):
    def __init__(self, backbone, head, policy):
        super().__init__() 
        self.network_class = Model
        self.backbone = backbone
        self.head = head
        self.policy = copy.deepcopy(policy)
        self.pos_networks = nn.ModuleList([policy])
        self.neg_networks = nn.ModuleList([])

    def forward(self, x):
        x, b_info = self.backbone(x)
        x, h_info = self.head(x)

        # Process positive networks
        pos_outs = torch.stack([network.forward(x)[0] for network in self.pos_networks], dim=0).sum(dim=0)

        if self.neg_networks:
            # Process negative networks
            neg_outs = torch.stack([network.forward(x)[0] for network in self.neg_networks], dim=0).sum(dim=0)
            q = pos_outs - neg_outs
        else:
            q = pos_outs

        info = {
            'backbone': b_info,
            'head': h_info,
            'policy': {}
        }
        return q, {}

    def plasticity_inject(self):
        pos_network = copy.deepcopy(self.policy)
        neg_network = copy.deepcopy(self.policy)
        self.pos_networks.append(pos_network)
        self.neg_networks.append(neg_network)
        # The only learnable network is the last positive network. 
        for network in (self.pos_networks[-2], self.neg_networks[-1]):
            for param in network.parameters():
                param.requires_grad = False
        
    def copy_online(self, online_model):
        for target_net, source_net in zip(self.pos_networks, online_model.pos_networks):
            target_net.load_state_dict(source_net.state_dict())
            for param in target_net.parameters():
                param.requires_grad = False
                
        for target_net, source_net in zip(self.neg_networks, online_model.neg_networks):
            target_net.load_state_dict(source_net.state_dict())
            for param in target_net.parameters():
                param.requires_grad = False