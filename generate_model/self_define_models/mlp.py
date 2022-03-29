import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim = 2):
        super(MLP, self).__init__()
        norm_layer = nn.BatchNorm1d
        self.input_fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    norm_layer(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4)
                )
        self.hidden_fcs = []
        for i in range(num_layers - 1):
            self.hidden_fcs.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    norm_layer(hidden_dim),
                    nn.ReLU(inplace=True)
                ))
        self.hidden_fcs = nn.ModuleList(self.hidden_fcs)
        self.fc_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_fc(x)
        for fc in self.hidden_fcs:
            x = fc(x)
        out = self.fc_layer(x)
        return out


      
