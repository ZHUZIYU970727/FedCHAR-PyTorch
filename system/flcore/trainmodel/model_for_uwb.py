import torch
import torch.nn.functional as F
from torch import nn
class BinaryClassifier(nn.Module):
    def __init__(self, input_size = 55, hidden_sizes = [32, 16]):
        super(BinaryClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.output = nn.Linear(prev_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.sigmoid(self.output(x))
        return x