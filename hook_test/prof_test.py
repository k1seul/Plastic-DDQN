import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# Define a simple neural network with a few layers for testing
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.layer5 = nn.Linear(100, 100)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# Hook function to save outputs into a dictionary
def save_outputs_hook(layer_id, layer_wise_outputs):
    def fn(_, __, output):
        layer_wise_outputs[layer_id] = output
    return fn

# Function to register hooks to the network
def register_hooks(net, num_hooks, layer_wise_outputs):
    # Register hooks on the first `num_hooks` layers
    hooks = []
    for idx, layer in enumerate(net.children()):
        if idx < num_hooks:
            hook = layer.register_forward_hook(save_outputs_hook(f"hook_{idx}", layer_wise_outputs))
            hooks.append(hook)
    return hooks

# Function to measure forward pass time with a varying number of hooks
def measure_time_with_hooks(net, max_hooks=5, hook_remove=False):
    input_tensor = torch.randn(1, 100)  # Random input tensor
    times = []
    
    # Dictionary to store layer-wise outputs
    layer_wise_outputs = {}

    for num_hooks in range(1, max_hooks + 1):
        # Register hooks
        hooks = register_hooks(net, num_hooks, layer_wise_outputs)
        
        # Measure forward pass time
        start_time = time.time()
        for _ in range(100):
            net(input_tensor)  # Forward pass
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # Remove hooks to avoid duplicates
        if hook_remove:
            for hook in hooks:
                hook.remove()

        # Clear the dictionary for the next iteration
        layer_wise_outputs.clear()

    return times

# Initialize the network and test the forward pass time with hooks
net = SimpleNet()
length = 1000
hook_remove = True
times = measure_time_with_hooks(net, max_hooks=length, hook_remove=hook_remove)

# Plot the computational complexity (time vs. number of hooks)
plt.plot(range(1, 1 + length), times, marker='o')
plt.xlabel('Number of Hooks')
plt.ylabel('Time (seconds)')
plt.title('Computational Complexity of Forward Pass w.r.t Hook Number')
plt.grid(True)
# plt.show()
plt.savefig(f"Computational Complexity of Forward Pass w.r.t Hook Number remove hook {str(hook_remove)}.png")
