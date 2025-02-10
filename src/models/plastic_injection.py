import copy
import torch
import torch.nn as nn
from .base import Model


class PlasticityInjectionModel(nn.Module):
    """
    A neural network model for plasticity injection, combining a backbone and a head 
    with a specified policy.

    Attributes:
        backbone (nn.Module): The feature extractor component of the model.
        head (nn.Module): The component that processes features extracted by the backbone.
        policy (nn.Module): A policy of the model, this is the only part affected by plasticity injection
    """
    def __init__(self, backbone, head, policy):
        """
        Initializes the PlasticityInjectionModel with a backbone, head, and policy.

        Args:
            backbone (nn.Module): The backbone of the model for feature extraction.
            head (nn.Module): The head of the model for processing features.
            policy (nn.Module): A policy of the model, this is the only part affected by plasticity injection
        """
        super().__init__() 
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
        self.pos_outs = pos_outs

        # Process negative networks
        if self.neg_networks:
            neg_outs = torch.stack([network.forward(x)[0] for network in self.neg_networks], dim=0).sum(dim=0)
            self.neg_outs = neg_outs
            q = pos_outs - neg_outs
        else:
            q = pos_outs

        info = {
            'backbone': b_info,
            'head': h_info,
            'policy': {}
        }
        return q, info

    def plasticity_inject(self):
        self.policy._initialize_weights()
        pos_network = copy.deepcopy(self.policy)
        neg_network = copy.deepcopy(self.policy)
        for pos, neg in zip(pos_network.parameters(), neg_network.parameters()):
            neg.data[:] = pos.data[:]
            neg.requires_grad = False
        self.pos_networks.append(pos_network)
        self.neg_networks.append(neg_network)
        # The only learnable network is the last positive network. 
        for param in self.pos_networks[-2].parameters():
            param.requires_grad = False
        
    def copy_online(self, online_model):
        """
        Copy parameters from other models that has been though plasticity injection.

        Args: online_model(nn.Module)
        """
        for online, target in zip(online_model.parameters(), self.parameters()):
            target.data[:] = online.data[:]
            target.requires_grad = False