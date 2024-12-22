
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class PPOAgent(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=128, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(features)
        return mean, std, value
