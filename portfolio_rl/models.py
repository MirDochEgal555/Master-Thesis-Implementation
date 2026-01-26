import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Outputs loc and log_std for logistic-normal / CLR sampling.
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
        self.loc_head = nn.Linear(hidden, K)
        self.logstd_head = nn.Linear(hidden, K)

    def forward(self, x):
        h = self.net(x)
        loc = self.loc_head(h)
        log_std = self.logstd_head(h).clamp(-3, 1)  # helps stability
        return loc, log_std


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

