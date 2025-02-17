from src.models import build_model
from hydra import compose, initialize
import numpy as np
import torch
from torch import optim

# Hydra Compose
initialize(version_base=None, config_path='configs') 
cfg = compose(config_name='ddqn_injection')
optimizer_cfg = cfg['agent']['optimizer']

# integrate hyper-params
param_dict = {'obs_shape': (4,1,84,84),
                'action_size': 8}
for key, value in param_dict.items():
        if key in cfg.model.backbone:
            cfg.model.backbone[key] = value
            
        if key in cfg.model.head:
            cfg.model.head[key] = value

        if key in cfg.model.policy:
            cfg.model.policy[key] = value
            
        if key in cfg.agent:
            cfg.agent[key] = value

# train a network with random inputs and outputs
def train_network(model):
    for _ in range(1000):
        test_input_state = torch.tensor(np.random.random((1, 1, 4, 1, 84, 84)), dtype=torch.float32)
        test_action_out = torch.tensor(np.random.random((1,8)), dtype=torch.float32)
        y, _ = model(test_input_state)
        mse_loss = loss(y, test_action_out)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
    
# Build ddqn_injection model
model = build_model(cfg.model)
target = build_model(cfg.model)
optimizer = optim.Adam(model.parameters(), 
                              **optimizer_cfg)
test_input_state = torch.tensor(np.random.random((1, 1, 4, 1, 84, 84)), dtype=torch.float32)
test_input_state_2 = torch.tensor(np.random.random((1, 1, 4, 1, 84, 84)), dtype=torch.float32)
test_action_out = torch.tensor(np.random.random((1,8)), dtype=torch.float32)
test_action_out_2 = torch.tensor(np.random.random((1,8)), dtype=torch.float32)
x, _ = model.backbone(test_input_state)
x, _ = model.head(x)

loss = torch.nn.MSELoss()

for _ in range(100):
    y, _ = model(test_input_state)
    mse_loss = loss(y, test_action_out)
    optimizer.zero_grad()
    mse_loss.backward()
    optimizer.step()

print(f'{model.pos_outs:}')
print(f'{mse_loss:}')
print(f'{y:}\n{test_action_out:}')
for online, tar in zip(model.parameters(), target.parameters()):
    tar.data[:] = online.data[:]
    tar.requires_grad = False
    
import pdb; pdb.set_trace()

for _ in range(100):
    y, _ = model(test_input_state_2)
    mse_loss = loss(y, test_action_out_2)
    optimizer.zero_grad()
    mse_loss.backward()
    optimizer.step()



print(f'{mse_loss:}')
print(f'{model.pos_outs:}\n{model.neg_outs:}')
print(f'{y:}\n{test_action_out_2:}')

# for online, tar in zip(model.parameters(), target.parameters()):
#     tar.data[:] = online.data[:]

for target_net, source_net in zip(target.pos_networks, model.pos_networks):
    target_net.load_state_dict(source_net.state_dict())
    for param in target_net.parameters():
                param.requires_grad = False

for target_net, source_net in zip(target.neg_networks, model.neg_networks):
    target_net.load_state_dict(source_net.state_dict())
    for param in target_net.parameters():
                param.requires_grad = False
import pdb; pdb.set_trace()

