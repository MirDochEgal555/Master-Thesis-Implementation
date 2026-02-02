import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Outputs concentration parameters for Dirichlet sampling.
    """
    def __init__(self, K: int, hidden: int = 128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.conc_head = nn.Linear(hidden, K)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.net(x)
        # Keep concentration strictly positive and mildly bounded away from 0.
        concentration = self.softplus(self.conc_head(h)) + 1e-3
        return concentration


class ValueNet(nn.Module):
    def __init__(self, K: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

