import torch
import torch.nn.functional as F

class QNetwork(torch.nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        NN_ARCHITECTURE = [
            {"input_dim": state_size, "output_dim": fc1_units, "activation": "relu"},
            {"input_dim": fc1_units, "output_dim": fc2_units, "activation": "relu"},
            {"input_dim": fc2_units, "output_dim": action_size, "activation": "sigmoid"},
        ]

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layers = []
        for idx, layer_arc in enumerate(NN_ARCHITECTURE):
            layer_idx = idx + 1
            layer_input_size = layer_arc["input_dim"]
            layer_output_size = layer_arc["output_dim"]
            self.layers.append(torch.nn.Linear(layer_input_size, layer_output_size))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)