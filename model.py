import torch
import torch.nn as nn
import torch.nn.functional as F
from prunable_layer import PrunableLinear

class SelfPruningNN(nn.Module):
    def __init__(self):
        super(SelfPruningNN, self).__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_all_gates(self):
        gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates.append(layer.get_gates())
        return gates